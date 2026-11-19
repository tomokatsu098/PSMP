"""
Unit and Integration Tests for PSMP 4-Phase Architecture (`test_psmp_phases.py`).
Verifies:
  - Phase 1: Historical data collection, coordinate POI clustering, stay duration distribution learning
  - Phase 2: IG-OFF initial power management plan & Jev (`choice`, `score`) decision schema
  - Phase 3: Parked monitoring, Jev (`noul`) wake-up trigger, Re-sleep on delay, SoC range guard
  - Phase 4: Post-trip evaluation, ROC curve & AUC computation, and threshold θ feedback tuning
"""

from datetime import datetime
import pytest

from psmp.config import DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import (
    BatteryThermalState,
    HVACState,
    IVIState,
    PowerPolicyMode,
    SentryState,
    VehicleSimulator,
)
from psmp.phase1_collector_learner import ParkingEventRecord, Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator


def test_jev_engine_system_one_primitives():
    """Jev ローカル判断特化型AIが choice / score / noul の3プリミティブと確率分布を1パスで返すことを検証"""
    engine = JevDecisionEngine(backend_mode="calibrated_local")
    state = {
        "elapsed_minutes": 40.0,
        "expected_mean_minutes": 50.0,
        "expected_std_minutes": 10.0,
        "device_lead_time_minutes": 12.0,
        "stat_return_prob_in_window": 0.68,
        "ambient_temp_c": 35.0,
        "cabin_temp_c": 39.0,
        "target_cabin_temp_c": 23.5,
        "has_passengers": True,
        "soc_pct": 65.0,
        "required_home_soc_pct": 25.0,
        "home_charge_priority_alpha": 0.3,
        "comfort_priority_beta": 0.85,
        "allow_enroute_charging": True,
    }
    questions = {
        "wake_up_hvac": {
            "type": "noul",
            "instructions": "Should HVAC wake up now?",
        },
        "policy": {
            "type": "choice",
            "instructions": "Select policy mode",
            "criteria": {
                "comfort_first": "Comfort priority",
                "balanced": "Balanced",
                "home_range_guard": "Eco priority",
            },
        },
        "intensity": {
            "type": "score",
            "instructions": "Precon intensity",
            "criteria": ["Lv0", "Lv1", "Lv2", "Lv3", "Lv4"],
        },
    }
    resp = engine.system_one(state, questions)
    assert "answers" in resp
    assert resp["usage"]["output_tokens"] == 0

    noul_ans = resp["answers"]["wake_up_hvac"]
    assert noul_ans["type"] == "noul"
    assert 0.50 < noul_ans["noul"] <= 1.0

    choice_ans = resp["answers"]["policy"]
    assert choice_ans["type"] == "choice"
    assert choice_ans["choice"] == "comfort_first"
    assert abs(sum(choice_ans["probabilities"].values()) - 1.0) < 1e-2

    score_ans = resp["answers"]["intensity"]
    assert score_ans["type"] == "score"
    assert 2.0 <= score_ans["score"] <= 4.0


def test_phase1_to_phase4_end_to_end():
    """Phase 1 〜 Phase 4 の一気通貫動作と ROC 感度評価・再調整フィードバックを検証"""
    p1 = Phase1CollectorLearner()
    records = p1.generate_synthetic_history(num_sessions_per_driver=40, seed=123)
    assert len(records) == 40 * len(DEFAULT_DRIVER_PROFILES)

    # 時系列分割: 前半を学習、後半をテスト
    train_recs = [r for i, r in enumerate(records) if i % 4 != 0]
    test_recs = [r for i, r in enumerate(records) if i % 4 == 0]

    p1.fit(train_recs)
    assert len(p1.learned_stats) > 0

    jev = JevDecisionEngine(backend_mode="calibrated_local")
    lgbm = LightGBMDecisionEngine(random_state=42)
    lgbm_summary = lgbm.fit(train_recs, p1)
    assert lgbm.is_fitted
    assert lgbm_summary.duration_rmse_min > 0.0

    sim = VehicleSimulator()
    p2 = Phase2IGOffPlanner(jev, sim)
    p3 = Phase3ParkedMonitor(p1, jev, sim, lgbm_engine=lgbm)
    p4 = Phase4PostTripEvaluator(p1, p2, p3)

    # 単一トリップの5手法比較 (EdgeJev vs LightGBM vs Stat Baseline vs Always-On vs No-Precon)
    sample_event = test_recs[0]
    comp = p4.evaluate_single_trip_comparison(sample_event)
    assert set(comp.keys()) == {"psmp_jev", "psmp_lightgbm", "stat_baseline", "always_on", "no_precon"}

    # PSMP (EdgeJev & LightGBM) は Always-On より消費電力が少なく、No-Precon より快適度が高いことを確認
    assert comp["psmp_jev"].parking_energy_consumed_wh <= comp["always_on"].parking_energy_consumed_wh
    assert comp["psmp_jev"].overall_ready_score >= comp["no_precon"].overall_ready_score

    # ROC曲線とAUC評価 (EdgeJev & LightGBM 両方の算出確認)
    report = p4.compute_roc_and_threshold_sweep(test_recs, theta_grid=[0.2, 0.4, 0.6, 0.8])
    assert report.roc_auc_jev >= 0.70
    assert report.roc_auc_lgbm >= 0.70
    assert len(report.operating_points) == 4
    assert len(report.operating_points_lgbm) == 4


    # 閾値 θ を上げたときに FPR と平均消費電力量が単調減少傾向にあることを確認
    op_low_theta = report.operating_points[0]   # theta = 0.2 (高感度)
    op_high_theta = report.operating_points[-1] # theta = 0.8 (省電力優先)
    assert op_low_theta.tpr_sensitivity >= op_high_theta.tpr_sensitivity
    assert op_low_theta.avg_energy_wh >= op_high_theta.avg_energy_wh

    # Phase 4 から Phase 1 への閾値再調整フィードバック
    updated_prof = p1.update_driver_preference("DRV_A_ECO_HOME", new_theta=0.68)
    assert updated_prof.roc_threshold_theta == 0.68


def test_unexpected_delay_resleep_and_soc_guard():
    """予定より大幅に遅延した際の自動再スリープ (Re-sleep) と SoC ガードの動作検証"""
    p1 = Phase1CollectorLearner()
    records = p1.generate_synthetic_history(num_sessions_per_driver=30, seed=99)
    p1.fit(records)

    jev = JevDecisionEngine()
    sim = VehicleSimulator()
    p2 = Phase2IGOffPlanner(jev, sim)
    p3 = Phase3ParkedMonitor(p1, jev, sim)

    # 予想約50分のスーパー駐車場で、実際には110分かかった大幅遅延イベントを作成
    poi = DEFAULT_POI_CLUSTERS["POI_SUPER_02"]
    delayed_event = ParkingEventRecord(
        session_id="SES_DELAY_TEST",
        driver_id="DRV_A_ECO_HOME",
        poi_id=poi.poi_id,
        lat=poi.lat,
        lon=poi.lon,
        park_start_dt=datetime(2026, 8, 10, 14, 0),
        departure_dt=datetime(2026, 8, 10, 15, 50),
        duration_minutes=110.0,
        day_of_week=0,
        is_holiday=False,
        hour_of_day=14,
        season="summer",
        ambient_temp_c=35.5,
        has_passengers=False,
        initial_soc_pct=38.0,
        distance_to_home_km=18.5,
        anomaly_type="unexpected_delay",
    )
    prof = p1.driver_profiles["DRV_A_ECO_HOME"]
    _, stay_dist = p1.predict_stay_distribution(
        delayed_event.driver_id, delayed_event.lat, delayed_event.lon,
        delayed_event.park_start_dt, delayed_event.is_holiday, delayed_event.has_passengers
    )
    plan, init_state = p2.create_initial_plan(delayed_event, prof, poi, stay_dist)
    res = p3.run_parked_session(delayed_event, prof, plan, init_state, override_theta=0.35)

    # 110分の間に一度事前起動した後、戻らないため Re-sleep が発動して無駄焚きを止めていることを確認
    assert any(log.re_sleep_triggered for log in res.timeline_logs)
    assert res.can_return_home_without_charging is True
