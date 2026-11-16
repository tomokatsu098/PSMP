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
