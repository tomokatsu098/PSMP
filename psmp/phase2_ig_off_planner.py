"""
Phase 2: IG-OFF Initial Power-Saving Planner (`phase2_ig_off_planner.py`)
２．駐車してIG-OFFしてから省電力マネジメントを実施するフェーズ
  - 駐車直後 (IG-OFF) に Phase 1 の学習済み滞在時間分布とドライバ嗜好を参照
  - 自宅までの帰還必要電力量と現在SoCから「駐車中に使える余剰電力バジェット」を算出
  - Jev ローカル判断特化型AI (`system_one`) を呼び出し、基本方針 (`PowerPolicyMode`)・
    Sentry監視レベル・初期スリープ状態を1パスで決定し、車載4機器を適切な省電力モードに移行させる
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from psmp.config import DEVICE_SPECS, DriverPreferenceProfile, POIClusterSpec
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.vehicle_devices import (
    BatteryThermalState,
    HVACState,
    IVIState,
    PowerPolicyMode,
    SentryState,
    VehicleEnvironmentState,
    VehicleSimulator,
)
from psmp.phase1_collector_learner import LearnedStayDistribution, ParkingEventRecord


@dataclass
class IGOffInitialPlan:
    """
    Phase 2 が IG-OFF 直後に策定する初期省電力マネジメント計画
    """
    session_id: str
    driver_id: str
    poi_spec: POIClusterSpec
    stay_distribution: LearnedStayDistribution

    # バッテリー＆帰宅エネルギー収支
    initial_soc_pct: float
    estimated_home_drive_kwh: float
    estimated_home_drive_soc_pct: float
    required_home_soc_with_margin_pct: float
    available_parking_budget_kwh: float
    available_parking_budget_soc_pct: float

    # Jev ローカル判断特化型AI の初期判定結果
    policy_mode: PowerPolicyMode
    policy_probabilities: Dict[str, float]
    policy_confidence: float
    initial_precon_intensity: float

    # 4機器の初期省電力移行状態
    initial_hvac_state: HVACState
    initial_ivi_state: IVIState
    initial_batt_state: BatteryThermalState
    initial_sentry_state: SentryState

    # 最初に起床監視を強化する予定時刻 (RTC Wakeup Target Minutes)
    scheduled_first_wakeup_check_min: float
    jev_raw_response: Dict[str, Any] = field(default_factory=dict)


class Phase2IGOffPlanner:
    """
    【Phase 2】駐車・IG-OFF トリガー時に起動し、自宅帰還電費計算と Jev 初期推論により
    車載4機器を適切な初期省電力モードへ移行させるモジュール
    """

    def __init__(
        self,
        jev_engine: JevDecisionEngine,
        vehicle_sim: VehicleSimulator,
    ):
        self.jev = jev_engine
        self.sim = vehicle_sim

    def create_initial_plan(
        self,
        event: ParkingEventRecord,
        profile: DriverPreferenceProfile,
        poi: POIClusterSpec,
        stay_dist: LearnedStayDistribution,
    ) -> Tuple[IGOffInitialPlan, VehicleEnvironmentState]:
        """
        IG-OFF 直後の車両状態・環境・学習分布から初期省電力プランを生成し、
        車載機器の初期状態スナップショット (`VehicleEnvironmentState`) を初期化する。
        """
        # 1. 自宅までの帰還電費と駐車中バジェット計算
        # 外気温が極端な場合は走行中の空調消費も増えるため 1.15倍の安全係数をかける
        temp_drive_factor = 1.15 if (event.ambient_temp_c >= 30.0 or event.ambient_temp_c <= 5.0) else 1.0
        est_home_drive_kwh = (event.distance_to_home_km / profile.efficiency_km_per_kwh) * temp_drive_factor
        est_home_drive_soc_pct = (est_home_drive_kwh / profile.battery_capacity_kwh) * 100.0

        req_home_soc_pct = est_home_drive_soc_pct + profile.min_home_arrival_soc_pct
        budget_soc_pct = max(0.0, event.initial_soc_pct - req_home_soc_pct)
        budget_kwh = (budget_soc_pct / 100.0) * profile.battery_capacity_kwh

        target_cabin_c = self.sim.compute_target_cabin_temp(event.ambient_temp_c)

        # 2. Jev ローカル判断特化型AI (`system_one`) への入力 State 構築
        jev_state = {
            "phase": "IG_OFF_INITIAL",
            "driver_id": event.driver_id,
            "poi_type": poi.poi_type.value,
            "expected_mean_minutes": stay_dist.mean_minutes,
            "expected_std_minutes": stay_dist.std_minutes,
            "ambient_temp_c": event.ambient_temp_c,
            "cabin_temp_c": target_cabin_c,  # IG-OFF直後は走行中空調により快適温度付近
            "target_cabin_temp_c": target_cabin_c,
            "has_passengers": event.has_passengers,
            "is_holiday": event.is_holiday,
            "soc_pct": event.initial_soc_pct,
            "required_home_soc_pct": round(req_home_soc_pct, 2),
            "home_charge_priority_alpha": profile.home_charge_priority_alpha,
            "comfort_priority_beta": profile.comfort_priority_beta,
            "allow_enroute_charging": profile.allow_enroute_charging,
            "security_risk_level": poi.security_risk_level,
        }

        jev_questions = {
            "power_policy_mode": {
                "type": "choice",
                "instructions": "Select the optimal vehicle power management policy upon IG-OFF.",
                "criteria": {
                    PowerPolicyMode.COMFORT_FIRST.value: "Prioritize cabin comfort and pre-conditioning even if en-route charging may be needed",
                    PowerPolicyMode.BALANCED.value: "Balance pre-conditioning comfort and battery energy conservation",
                    PowerPolicyMode.HOME_RANGE_GUARD.value: "Strictly preserve battery SoC to guarantee arriving home without charging",
                    PowerPolicyMode.EMERGENCY_SAVE.value: "Critical low SoC cutoff: disable all non-essential power drains immediately",
                },
            },
            "sentry_mode_choice": {
                "type": "choice",
                "instructions": "Select security camera monitoring mode during parking.",
                "criteria": {
                    "full_ai_sentry": "High security risk and sufficient battery: keep 4-camera AI sentry active",
                    "low_pir_sentry": "Moderate/low risk or battery conservation needed: use low-power PIR sensor trigger",
                },
            },
            "precon_intensity_level": {
                "type": "score",
                "instructions": "Evaluate desired pre-conditioning power intensity level (0=minimal to 4=maximum).",
                "criteria": [
                    "Minimal power (strictly eco)",
                    "Low power pre-conditioning",
                    "Moderate standard pre-conditioning",
                    "High power rapid pre-conditioning",
                    "Maximum boost for extreme weather or passengers",
                ],
            },
        }

        jev_resp = self.jev.system_one(jev_state, jev_questions)
        ans_policy = jev_resp["answers"]["power_policy_mode"]
        ans_sentry = jev_resp["answers"]["sentry_mode_choice"]
        ans_score = jev_resp["answers"]["precon_intensity_level"]

        policy_mode = PowerPolicyMode(ans_policy["choice"])
        # Score (0~4) を 0.45 ~ 1.0 の出力強度係数へ変換
        norm_intensity = round(0.45 + 0.55 * (float(ans_score["score"]) / 4.0), 3)

        # 3. 予想滞在時間と Policy Mode に基づく4機器の初期省電力モード決定
        is_very_short_stay = stay_dist.mean_minutes <= 16.0
        is_extreme_temp = (event.ambient_temp_c >= 31.0) or (event.ambient_temp_c <= 5.0)

        # ① 空調 (HVAC): コンビニ等の超短時間駐車 かつ 猛暑/厳冬 かつ バジェット余裕ありなら ECO_HOLD で室温維持、それ以外は OFF
        if (
            is_very_short_stay
            and is_extreme_temp
            and policy_mode in (PowerPolicyMode.COMFORT_FIRST, PowerPolicyMode.BALANCED)
            and budget_soc_pct >= 4.0
        ):
            init_hvac = HVACState.ECO_HOLD
        else:
            init_hvac = HVACState.OFF

        # ② IVI / ナビ / 通信: 20分以内の短時間駐車なら STANDBY (RAM保持)、それ以外は DEEP_SLEEP
        if stay_dist.mean_minutes <= 20.0 and policy_mode != PowerPolicyMode.EMERGENCY_SAVE:
            init_ivi = IVIState.STANDBY
        else:
            init_ivi = IVIState.DEEP_SLEEP

        # ③ 駆動用バッテリー温調: IG-OFF直後は走行直後で温まっているため一旦 OFF へ移行
        init_batt = BatteryThermalState.OFF

        # ④ 監視カメラ (Sentry): Jev判定とSoCガードに基づき決定
        if policy_mode in (PowerPolicyMode.HOME_RANGE_GUARD, PowerPolicyMode.EMERGENCY_SAVE):
            init_sentry = SentryState.LOW_POWER_PIR
        elif ans_sentry["choice"] == "full_ai_sentry" and budget_soc_pct >= 6.0:
            init_sentry = SentryState.FULL_AI_SENTRY
        else:
            init_sentry = SentryState.LOW_POWER_PIR

        # 初回起床チェック予定時刻 (予想滞在時間の p10 - 最大リードタイム、最低でも2分後)
        max_lead = max(DEVICE_SPECS.hvac_lead_time_minutes, DEVICE_SPECS.batt_lead_time_minutes)
        first_wakeup_min = max(2.0, round(stay_dist.p10_minutes - max_lead - 4.0, 1))

        # 初期状態スナップショットの生成 (IG-OFF直後の車内温度は快適温度+0.5℃、バッテリー温度は22℃からスタート)
        initial_env_state = VehicleEnvironmentState(
            elapsed_minutes=0.0,
            ambient_temp_c=event.ambient_temp_c,
            cabin_temp_c=round(target_cabin_c + (0.8 if event.ambient_temp_c > target_cabin_c else -0.8), 2),
            target_cabin_temp_c=target_cabin_c,
            battery_temp_c=21.0 if event.ambient_temp_c >= 10.0 else 15.0,
            soc_pct=event.initial_soc_pct,
            battery_energy_kwh=round((event.initial_soc_pct / 100.0) * profile.battery_capacity_kwh, 4),
            battery_capacity_kwh=profile.battery_capacity_kwh,
            hvac_state=init_hvac,
            ivi_state=init_ivi,
            batt_thermal_state=init_batt,
            sentry_state=init_sentry,
            precon_intensity=norm_intensity,
        )

        plan = IGOffInitialPlan(
            session_id=event.session_id,
            driver_id=event.driver_id,
            poi_spec=poi,
            stay_distribution=stay_dist,
            initial_soc_pct=event.initial_soc_pct,
            estimated_home_drive_kwh=round(est_home_drive_kwh, 3),
            estimated_home_drive_soc_pct=round(est_home_drive_soc_pct, 2),
            required_home_soc_with_margin_pct=round(req_home_soc_pct, 2),
            available_parking_budget_kwh=round(budget_kwh, 3),
            available_parking_budget_soc_pct=round(budget_soc_pct, 2),
            policy_mode=policy_mode,
            policy_probabilities=ans_policy["probabilities"],
            policy_confidence=ans_policy["confidence"],
            initial_precon_intensity=norm_intensity,
            initial_hvac_state=init_hvac,
            initial_ivi_state=init_ivi,
            initial_batt_state=init_batt,
            initial_sentry_state=init_sentry,
            scheduled_first_wakeup_check_min=first_wakeup_min,
            jev_raw_response=jev_resp,
        )
        return plan, initial_env_state
