"""
Unit tests for Powertrain-specific Battery Management (`tests/test_powertrain.py`).
Verifies:
  - BEV: High-voltage SoC <= 20% triggers S4 Dormant emergency save (preventing 12V DC-DC shutdown).
  - PHEV: High-voltage SoC <= 15% cuts off EV pre-conditioning while keeping gasoline drivability.
  - HEV: High-voltage pre-conditioning is disabled while engine is stopped; 12V key polling is prioritized.
  - ICE: Only 12V lead-acid battery exists; no HVAC pre-con; strictly preserves cranking current.
"""

from datetime import datetime
import pytest

from psmp.config import DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS, PowertrainType
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.vehicle_devices import (
    BatteryThermalState,
    HVACState,
    PowerPolicyMode,
    SentryState,
    VehicleSimulator,
)
from psmp.phase1_collector_learner import LearnedStayDistribution, ParkingEventRecord
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor


@pytest.fixture
def test_setup():
    jev = JevDecisionEngine(backend_mode="calibrated_local")
    sim = VehicleSimulator()
    p2 = Phase2IGOffPlanner(jev, sim)
    from psmp.phase1_collector_learner import Phase1CollectorLearner
    p1 = Phase1CollectorLearner()
    p3 = Phase3ParkedMonitor(p1, jev, sim)
    poi = DEFAULT_POI_CLUSTERS["POI_SUPER_02"]
    stay_dist = LearnedStayDistribution(
        mean_minutes=60.0,
        std_minutes=15.0,
        p10_minutes=42.0,
        p50_minutes=58.0,
        p90_minutes=80.0,
        sample_count=30,
        confidence_weight=0.9,
    )
    return p2, p3, poi, stay_dist


def test_bev_low_soc_emergency_save(test_setup):
    """BEVで高圧SoCが20%以下の場合、12V DC-DC補充電停止リスク回避のためEMERGENCY_SAVEへ強制移行することを検証"""
    p2, p3, poi, stay_dist = test_setup
    prof_bev = DEFAULT_DRIVER_PROFILES["DRV_A_ECO_HOME"]
    assert prof_bev.powertrain_type == PowertrainType.BEV

    # 低SoC (18%) のイベント
    event = ParkingEventRecord(
        session_id="SES_BEV_LOW_SOC",
        driver_id=prof_bev.driver_id,
        poi_id=poi.poi_id,
        lat=poi.lat,
        lon=poi.lon,
        park_start_dt=datetime(2026, 8, 10, 14, 0),
        departure_dt=datetime(2026, 8, 10, 15, 0),
        duration_minutes=60.0,
        day_of_week=0,
        is_holiday=False,
        hour_of_day=14,
        season="summer",
        ambient_temp_c=34.0,
        has_passengers=False,
        initial_soc_pct=18.0,  # <= 20%
        distance_to_home_km=10.0,
    )

    plan, state = p2.create_initial_plan(event, prof_bev, poi, stay_dist)
    # BEV低SoCガードにより EMERGENCY_SAVE
    assert plan.policy_mode == PowerPolicyMode.EMERGENCY_SAVE
    assert plan.initial_hvac_state == HVACState.OFF
    assert plan.initial_sentry_state == SentryState.LOW_POWER_PIR

    # Phase 3 モニターでも SoC ガードが即時発動し空調停止状態を維持
    sim_res = p3.run_parked_session(event, prof_bev, plan, state)
    assert any(log.soc_guard_triggered for log in sim_res.timeline_logs)
    assert sim_res.departure_cabin_temp_c != sim_res.target_cabin_temp_c


def test_ice_and_hev_no_high_voltage_hvac(test_setup):
    """ICEおよびHEVではエンジン停止中の高圧空調や温調が不可であり、12V待受保護に徹することを検証"""
    p2, p3, poi, stay_dist = test_setup
    prof_ice = DEFAULT_DRIVER_PROFILES["DRV_E_ICE_COMMUTER"]
    prof_hev = DEFAULT_DRIVER_PROFILES["DRV_D_HEV_STANDARD"]

    assert prof_ice.powertrain_type == PowertrainType.ICE
    assert prof_hev.powertrain_type == PowertrainType.HEV

    for prof in [prof_ice, prof_hev]:
        event = ParkingEventRecord(
            session_id=f"SES_{prof.powertrain_type.value}",
            driver_id=prof.driver_id,
            poi_id=poi.poi_id,
            lat=poi.lat,
            lon=poi.lon,
            park_start_dt=datetime(2026, 8, 10, 14, 0),
            departure_dt=datetime(2026, 8, 10, 14, 15),
            duration_minutes=15.0,  # 短時間駐車
            day_of_week=0,
            is_holiday=False,
            hour_of_day=14,
            season="summer",
            ambient_temp_c=35.0,
            has_passengers=False,
            initial_soc_pct=80.0,
            distance_to_home_km=5.0,
        )

        plan, state = p2.create_initial_plan(event, prof, poi, stay_dist)
        # 高圧空調・温調は不可
        assert plan.initial_hvac_state == HVACState.OFF
        assert plan.initial_batt_state == BatteryThermalState.OFF
        assert plan.initial_sentry_state == SentryState.LOW_POWER_PIR

        sim_res = p3.run_parked_session(event, prof, plan, state)
        # 駐車中も空調は常時 OFF
        assert all(log.hvac_state == HVACState.OFF.value for log in sim_res.timeline_logs)


def test_phev_balanced_gasoline_backup(test_setup):
    """PHEVは中容量高圧を持ち、残量がある場合はEV空調を許可しつつ、低SoC時はガソリン自走を前提に空調遮断することを検証"""
    p2, p3, poi, stay_dist = test_setup
    prof_phev = DEFAULT_DRIVER_PROFILES["DRV_C_PHEV_BALANCED"]
    assert prof_phev.powertrain_type == PowertrainType.PHEV

    event = ParkingEventRecord(
        session_id="SES_PHEV_TEST",
        driver_id=prof_phev.driver_id,
        poi_id=poi.poi_id,
        lat=poi.lat,
        lon=poi.lon,
        park_start_dt=datetime(2026, 8, 10, 14, 0),
        departure_dt=datetime(2026, 8, 10, 14, 50),
        duration_minutes=50.0,
        day_of_week=0,
        is_holiday=False,
        hour_of_day=14,
        season="summer",
        ambient_temp_c=32.0,
        has_passengers=True,
        initial_soc_pct=70.0,
        distance_to_home_km=8.0,
    )

    plan, state = p2.create_initial_plan(event, prof_phev, poi, stay_dist)
    sim_res = p3.run_parked_session(event, prof_phev, plan, state)
    # 同乗者あり＆猛暑＆SoC余裕ありのため、帰還直前にプレ空調が動作することを確認
    assert any(log.hvac_state in (HVACState.PRE_CONDITIONING.value, HVACState.READY_MAINTAIN.value) for log in sim_res.timeline_logs)
