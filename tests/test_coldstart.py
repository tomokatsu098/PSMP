"""
Unit tests for PSMP Cold-start / Unvisited POI handling with Driver Category ECDF.
"""
from datetime import datetime
import numpy as np
import pytest

from psmp.config import DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS, POIClusterSpec, POIType
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import ParkingEventRecord, Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator


def test_driver_category_ecdf_coldstart():
    p1 = Phase1CollectorLearner()
    records = p1.generate_synthetic_history(num_sessions_per_driver=50, seed=42)
    p1.fit(records)

    # ドライバ別カテゴリECDFと全出先滞在時間が学習されていることを検証
    assert len(p1.driver_category_durations) > 0
    assert len(p1.driver_global_durations) > 0
    for drv_id in p1.driver_profiles.keys():
        assert drv_id in p1.driver_global_durations
        assert len(p1.driver_global_durations[drv_id]) > 0

    # 未登録の初訪問POI（Cold-start）を定義
    unvisited_poi = POIClusterSpec(
        poi_id="POI_UNKNOWN_NEW_01",
        name_ja="初めて行くカフェ",
        poi_type=POIType.CONVENIENCE_STORE,
        lat=35.6100,
        lon=139.7100,
        distance_to_home_km=20.0,
        security_risk_level=0.3,
        base_mean_minutes=20.0,
        base_std_minutes=5.0,
    )
    p1.poi_clusters[unvisited_poi.poi_id] = unvisited_poi

    # 未学習POIに対して predict_stay_distribution を呼び出す
    poi, stay_dist = p1.predict_stay_distribution(
        driver_id="DRV_A_ECO_HOME",
        lat=35.6100,
        lon=139.7100,
        park_start_dt=datetime(2026, 7, 10, 14, 0),
        is_holiday=False,
        has_passengers=False,
    )

    # ドライバ別カテゴリECDFが自動適用され、raw_durations が設定されていることを検証
    assert poi.poi_id == "POI_UNKNOWN_NEW_01"
    assert stay_dist.confidence_weight >= 0.70
    assert stay_dist.raw_durations is not None
    assert len(stay_dist.raw_durations) >= 2

    # 条件付き帰還確率が計算できること
    prob_10min = p1.compute_conditional_return_probability(
        stay_dist, elapsed_minutes=5.0, horizon_minutes=10.0
    )
    assert 0.005 <= prob_10min <= 0.995

    # Phase 2 〜 Phase 3 での動作確認
    jev = JevDecisionEngine(backend_mode="calibrated_local")
    sim = VehicleSimulator()
    p2 = Phase2IGOffPlanner(jev, sim)
    p3 = Phase3ParkedMonitor(p1, jev, sim)

    test_event = ParkingEventRecord(
        session_id="SES_COLD_001",
        driver_id="DRV_A_ECO_HOME",
        poi_id="POI_UNKNOWN_NEW_01",
        lat=35.6100,
        lon=139.7100,
        park_start_dt=datetime(2026, 7, 10, 14, 0),
        departure_dt=datetime(2026, 7, 10, 14, 25),
        duration_minutes=25.0,
        day_of_week=4,
        is_holiday=False,
        hour_of_day=14,
        season="summer",
        ambient_temp_c=33.0,
        has_passengers=False,
        initial_soc_pct=65.0,
        distance_to_home_km=20.0,
    )

    prof = p1.driver_profiles["DRV_A_ECO_HOME"]
    plan, state = p2.create_initial_plan(test_event, prof, poi, stay_dist)
    res = p3.run_parked_session(test_event, prof, plan, state)

    assert res.actual_duration_minutes == 25.0
    assert res.parking_energy_consumed_wh > 0.0
    assert len(res.timeline_logs) > 0


def test_home_vs_out_category_separation():
    """自宅駐車 (HOME_OVERNIGHT / HOME_LONG) と出先駐車 (QUICK / MEDIUM / LONG) の分別ECDF学習検証"""
    p1 = Phase1CollectorLearner()

    records = [
        # 出先短時間（コンビニ）
        ParkingEventRecord(
            session_id="SES_OUT_01",
            driver_id="DRV_A_ECO_HOME",
            poi_id="POI_CONV_01",
            lat=35.6812,
            lon=139.7671,
            park_start_dt=datetime(2026, 7, 1, 9, 0),
            departure_dt=datetime(2026, 7, 1, 9, 15),
            duration_minutes=15.0,
            day_of_week=2,
            is_holiday=False,
            hour_of_day=9,
            season="summer",
            ambient_temp_c=28.0,
            has_passengers=False,
            initial_soc_pct=70.0,
            distance_to_home_km=12.0,
            is_home=False,
        ),
        # 出先中時間（スーパー）
        ParkingEventRecord(
            session_id="SES_OUT_02",
            driver_id="DRV_A_ECO_HOME",
            poi_id="POI_SUPER_02",
            lat=35.6580,
            lon=139.7515,
            park_start_dt=datetime(2026, 7, 1, 14, 0),
            departure_dt=datetime(2026, 7, 1, 14, 50),
            duration_minutes=50.0,
            day_of_week=2,
            is_holiday=False,
            hour_of_day=14,
            season="summer",
            ambient_temp_c=31.0,
            has_passengers=False,
            initial_soc_pct=65.0,
            distance_to_home_km=18.5,
            is_home=False,
        ),
        # 自宅日常夜間（OVERNIGHT: 12時間 = 720分）
        ParkingEventRecord(
            session_id="SES_HOME_01",
            driver_id="DRV_A_ECO_HOME",
            poi_id="POI_HOME_00",
            lat=35.6800,
            lon=139.7600,
            park_start_dt=datetime(2026, 7, 1, 20, 0),
            departure_dt=datetime(2026, 7, 2, 8, 0),
            duration_minutes=720.0,
            day_of_week=2,
            is_holiday=False,
            hour_of_day=20,
            season="summer",
            ambient_temp_c=24.0,
            has_passengers=False,
            initial_soc_pct=50.0,
            distance_to_home_km=0.0,
            is_home=True,
        ),
        # 自宅長期放置（HOME_LONG: 3日 = 4320分）
        ParkingEventRecord(
            session_id="SES_HOME_02",
            driver_id="DRV_A_ECO_HOME",
            poi_id="POI_HOME_00",
            lat=35.6800,
            lon=139.7600,
            park_start_dt=datetime(2026, 7, 3, 18, 0),
            departure_dt=datetime(2026, 7, 6, 18, 0),
            duration_minutes=4320.0,
            day_of_week=4,
            is_holiday=False,
            hour_of_day=18,
            season="summer",
            ambient_temp_c=25.0,
            has_passengers=False,
            initial_soc_pct=85.0,
            distance_to_home_km=0.0,
            is_home=True,
        ),
    ]

    p1.fit(records)

    # 1. カテゴリ別に正しく分離されて登録されていること
    assert ("DRV_A_ECO_HOME", "QUICK") in p1.driver_category_durations
    assert ("DRV_A_ECO_HOME", "MEDIUM") in p1.driver_category_durations
    assert ("DRV_A_ECO_HOME", "HOME_OVERNIGHT") in p1.driver_category_durations
    assert ("DRV_A_ECO_HOME", "HOME_LONG") in p1.driver_category_durations

    # 2. 出先ECDFに自宅の720分や4320分が混入していないこと
    quick_durs = p1.driver_category_durations[("DRV_A_ECO_HOME", "QUICK")]
    assert len(quick_durs) == 1 and quick_durs[0] == 15.0

    med_durs = p1.driver_category_durations[("DRV_A_ECO_HOME", "MEDIUM")]
    assert len(med_durs) == 1 and med_durs[0] == 50.0

    # 3. 自宅ECDFに正しく格納されていること
    home_ovn_durs = p1.driver_category_durations[("DRV_A_ECO_HOME", "HOME_OVERNIGHT")]
    assert len(home_ovn_durs) == 1 and home_ovn_durs[0] == 720.0

    home_long_durs = p1.driver_category_durations[("DRV_A_ECO_HOME", "HOME_LONG")]
    assert len(home_long_durs) == 1 and home_long_durs[0] == 4320.0

    # 4. 出先全体プール (driver_out_durations) と自宅全体プール (driver_home_durations) の分離
    assert len(p1.driver_out_durations["DRV_A_ECO_HOME"]) == 2
    assert max(p1.driver_out_durations["DRV_A_ECO_HOME"]) == 50.0

    assert len(p1.driver_home_durations["DRV_A_ECO_HOME"]) == 2
    assert min(p1.driver_home_durations["DRV_A_ECO_HOME"]) == 720.0
