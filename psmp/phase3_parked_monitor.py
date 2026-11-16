"""
Phase 3: Parked Monitor & Dynamic Power-Saving Controller (`phase3_parked_monitor.py`)
３．駐車中、ドライバが戻ってくるまで監視・待機・省電力モード調整するフェーズ
  - 駐車中、ドライバが戻ってくる (IG-ON) までタイムステップごとに環境・SoC・経過時間を監視
  - Jev ローカル判断特化型AI (`edgejev` System One: `noul` / `score`) および
    LightGBM (`LGBMRegressor` + `LGBMClassifier`) の両方または一方を用いて、
    各機器の必要リードタイム以内にドライバが帰還する確率 P(wake) を推論
  - ROC動作点閾値 θ と比較して車載機器 (HVAC, 駆動用バッテリー温調, IVI, Sentry) を段階的にレディ状態へ復帰
  - 「時に…」の動的例外対応:
    1) 予定遅延時の自動 Re-sleep 制御（最大ホールド時間を超えても戻らない場合は無駄焚きを止めて省電力待機へ戻す）
    2) SoC 下限安全マージン到達時の自宅帰還優先カットオフ（個人嗜好が無充電帰宅派の場合に空調等を抑制）
    3) 猛暑・厳冬・同乗者あり時の動的閾値補正
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from psmp.config import DEVICE_SPECS, DeviceHardwareSpec, DriverPreferenceProfile
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import (
    BatteryThermalState,
    HVACState,
    IVIState,
    PowerPolicyMode,
    SentryState,
    VehicleEnvironmentState,
    VehicleSimulator,
)
from psmp.phase1_collector_learner import ParkingEventRecord, Phase1CollectorLearner
from psmp.phase2_ig_off_planner import IGOffInitialPlan


@dataclass
class TimeStepLog:
    """
    駐車中の各タイムステップ t における監視・EdgeJev推論・LightGBM推論・機器状態・ROC真偽ラベルの記録
    """
    step_index: int
    elapsed_minutes: float
    remaining_minutes_to_departure: float

    # 統計ハザード & EdgeJev & LightGBM 推論確率 (0.0 ~ 1.0)
    stat_hazard_prob: float
    jev_wake_hvac_prob: float
    lgbm_wake_hvac_prob: float
    active_wake_hvac_prob: float
    jev_wake_ivi_prob: float
    jev_wake_batt_prob: float
    effective_threshold_theta: float

    # ROC評価用の正解ラベル (Ground Truth):
    # 「本当に今後 HVAC リードタイム + 許容窓 (例: 16分) 以内にドライバが戻ってきたか？」
    ground_truth_should_wake_hvac: bool
    # 実際に予測モデルが HVAC 事前稼働トリガーを ON にしたか
    predicted_wake_hvac: bool

    # 車載4機器の状態と環境値
    hvac_state: str
    ivi_state: str
    batt_thermal_state: str
    sentry_state: str
    policy_mode: str
    cabin_temp_c: float
    ambient_temp_c: float
    battery_temp_c: float
    soc_pct: float
    step_power_w: float
    cumulative_energy_wh: float
    re_sleep_triggered: bool = False
    soc_guard_triggered: bool = False


@dataclass
class ParkingSessionSimulationResult:
    """
    1回の駐車セッション（IG-OFF 〜 Phase 2 〜 Phase 3 〜 IG-ON 発進 〜 帰宅）の総合シミュレーション結果
    """
    session_id: str
    driver_id: str
    poi_id: str
    poi_name_ja: str
    anomaly_type: str
    has_passengers: bool
    is_holiday: bool
    season: str
    ambient_temp_c: float
    actual_duration_minutes: float
    expected_mean_minutes: float

    # 制御モード・設定閾値
    control_strategy: str   # "psmp_jev" | "psmp_lightgbm" | "stat_baseline" | "always_on" | "no_precon"
    roc_threshold_theta: float
    initial_policy_mode: str

    # 入車時 (IG-ON) の状態・快適度評価
    departure_cabin_temp_c: float
    target_cabin_temp_c: float
    departure_temp_error_c: float
    cabin_comfort_score: float
    ivi_ready_score: float
    batt_ready_score: float
    overall_ready_score: float
    is_ready_success: bool

    # エネルギー・バッテリーSoC・帰宅成否評価
    initial_soc_pct: float
    departure_soc_pct: float
    parking_energy_consumed_wh: float
    estimated_home_arrival_soc_pct: float
    min_home_arrival_soc_pct: float
    can_return_home_without_charging: bool
    enroute_charging_required: bool

    # 時系列ログ
    timeline_logs: List[TimeStepLog] = field(default_factory=list)


class Phase3ParkedMonitor:
    """
    【Phase 3】駐車中の監視・待機・EdgeJev / LightGBM 動的確率推論・機器レディ復帰・再スリープ制御モジュール
    """

    def __init__(
        self,
        phase1_learner: Phase1CollectorLearner,
        jev_engine: JevDecisionEngine,
        vehicle_sim: VehicleSimulator,
        lgbm_engine: Optional[LightGBMDecisionEngine] = None,
        specs: DeviceHardwareSpec = DEVICE_SPECS,
    ):
        self.learner = phase1_learner
        self.jev = jev_engine
        self.lgbm = lgbm_engine
        self.sim = vehicle_sim
        self.specs = specs

    def run_parked_session(
        self,
        event: ParkingEventRecord,
        profile: DriverPreferenceProfile,
        initial_plan: IGOffInitialPlan,
        initial_env_state: VehicleEnvironmentState,
        dt_minutes: float = 2.0,
        override_theta: Optional[float] = None,
        control_strategy: str = "psmp_jev",
    ) -> ParkingSessionSimulationResult:
        """
        IG-OFF (t=0) からドライバ帰還 IG-ON (t=event.duration_minutes) まで
        `dt_minutes` 刻みで監視・制御ループを回し、結果を返す。

        Args:
            control_strategy:
                - "psmp_jev": 提案手法①（EdgeJev ローカル判断特化型AI + ROC閾値制御 + Re-sleep + SoCガード）
                - "psmp_lightgbm": 比較手法②（LightGBM GBDTモデル + ROC閾値制御 + Re-sleep + SoCガード）
                - "stat_baseline": 比較手法③（単純POI統計タイマー方式）
                - "always_on": 常時快適維持（駐車中ずっと空調・IVI・Sentryを稼働させる比較用上限）
                - "no_precon": 完全省電力放置（事前復帰を一切行わない比較用下限）
        """
        state = deepcopy(initial_env_state)
        base_theta = override_theta if override_theta is not None else profile.roc_threshold_theta
        current_policy = initial_plan.policy_mode

        timeline: List[TimeStepLog] = []
        total_duration = event.duration_minutes
        step_idx = 0
        re_sleep_cooldown_until = -1.0  # 再スリープ後のクールダウンタイマー

        # LightGBM が有効な場合はセッション滞在時間予測を事前キャッシュ
        cached_lgbm_stay = None
        if self.lgbm is not None and self.lgbm.is_fitted:
            cached_lgbm_stay = self.lgbm.predict_stay_duration(event)

        while state.elapsed_minutes < total_duration:
            t = state.elapsed_minutes
            rem_min = max(0.0, total_duration - t)
            step_dt = min(dt_minutes, rem_min)

            # 1. Phase 1 の統計モデルから条件付き帰還確率を計算
            cond_prob_hvac = self.learner.compute_conditional_return_probability(
                initial_plan.stay_distribution,
                elapsed_minutes=t,
                horizon_minutes=self.specs.hvac_lead_time_minutes,
            )
            coarse_stat_prob = self.learner.compute_coarse_poi_return_probability(
                initial_plan.poi_spec,
                elapsed_minutes=t,
                horizon_minutes=self.specs.hvac_lead_time_minutes,
            )
            stat_prob_ivi = self.learner.compute_conditional_return_probability(
                initial_plan.stay_distribution,
                elapsed_minutes=t,
                horizon_minutes=self.specs.ivi_lead_time_minutes + 2.0,
            )
            stat_prob_batt = self.learner.compute_conditional_return_probability(
                initial_plan.stay_distribution,
                elapsed_minutes=t,
                horizon_minutes=self.specs.batt_lead_time_minutes,
            )

            # 2A. EdgeJev ローカル判断特化型AI (`system_one`) による動的1-Pass推論
            jev_state_hvac = {
                "phase": "PARKED_MONITOR",
                "elapsed_minutes": t,
                "expected_mean_minutes": initial_plan.stay_distribution.mean_minutes,
                "expected_std_minutes": initial_plan.stay_distribution.std_minutes,
                "device_lead_time_minutes": self.specs.hvac_lead_time_minutes,
                "stat_return_prob_in_window": cond_prob_hvac,
                "ambient_temp_c": state.ambient_temp_c,
                "cabin_temp_c": state.cabin_temp_c,
                "target_cabin_temp_c": state.target_cabin_temp_c,
                "has_passengers": event.has_passengers,
                "is_holiday": event.is_holiday,
                "soc_pct": state.soc_pct,
                "required_home_soc_pct": initial_plan.required_home_soc_with_margin_pct,
                "home_charge_priority_alpha": profile.home_charge_priority_alpha,
                "comfort_priority_beta": profile.comfort_priority_beta,
                "allow_enroute_charging": profile.allow_enroute_charging,
            }

            jev_questions = {
                "wake_up_hvac": {
                    "type": "noul",
                    "instructions": "Should HVAC pre-conditioning wake up now so the cabin is comfortable when the driver returns?",
                },
                "dynamic_precon_intensity": {
                    "type": "score",
                    "instructions": "Dynamic pre-conditioning power level based on current cabin temperature gap and SoC margin.",
                    "criteria": ["Eco minimal", "Low", "Medium", "High", "Max boost"],
                },
            }

            jev_resp = self.jev.system_one(jev_state_hvac, jev_questions)
            jev_prob_hvac = float(jev_resp["answers"]["wake_up_hvac"]["noul"])
            dyn_score = float(jev_resp["answers"]["dynamic_precon_intensity"]["score"])
            state.precon_intensity = round(0.45 + 0.55 * (dyn_score / 4.0), 3)

            # 2B. LightGBM による動的復帰確率推論
            soc_margin = state.soc_pct - initial_plan.required_home_soc_with_margin_pct
            if self.lgbm is not None and self.lgbm.is_fitted:
                lgbm_prob_hvac, _ = self.lgbm.predict_wakeup_probability(
                    rec=event,
                    elapsed_minutes=t,
                    expected_std_minutes=initial_plan.stay_distribution.std_minutes,
                    stat_cond_hazard_prob=cond_prob_hvac,
                    soc_margin_pct=soc_margin,
                    alpha_range=profile.home_charge_priority_alpha,
                    beta_comfort=profile.comfort_priority_beta,
                    cached_pred_stay=cached_lgbm_stay,
                )
            else:
                lgbm_prob_hvac = cond_prob_hvac

            # 3. 制御戦略に応じたアクティブ予測確率と実効閾値 eff_theta の決定
            soc_guard_triggered = False
            re_sleep_triggered = False
            eff_theta = base_theta

            if control_strategy == "psmp_jev":
                active_prob_hvac = jev_prob_hvac
                # 猛暑・厳冬または同乗者ありで、かつバッテリーに余裕がある場合は実効閾値θを下げて高感度化
                is_harsh_weather = (state.ambient_temp_c >= 31.0) or (state.ambient_temp_c <= 5.0)
                if (event.has_passengers or is_harsh_weather) and (soc_margin >= 6.0 or profile.allow_enroute_charging):
                    eff_theta = max(0.08, base_theta - 0.08)
                # 自宅無充電帰還こだわるドライバでSoCマージンが残り少ない場合は閾値θを引き上げ（無駄焚き防止）
                if soc_margin <= 2.0 and not profile.allow_enroute_charging:
                    soc_guard_triggered = True
                    current_policy = PowerPolicyMode.HOME_RANGE_GUARD
                    eff_theta = min(0.92, base_theta + 0.25)

            elif control_strategy == "psmp_lightgbm":
                active_prob_hvac = lgbm_prob_hvac
                is_harsh_weather = (state.ambient_temp_c >= 31.0) or (state.ambient_temp_c <= 5.0)
                if (event.has_passengers or is_harsh_weather) and (soc_margin >= 6.0 or profile.allow_enroute_charging):
                    eff_theta = max(0.08, base_theta - 0.08)
                if soc_margin <= 2.0 and not profile.allow_enroute_charging:
                    soc_guard_triggered = True
                    current_policy = PowerPolicyMode.HOME_RANGE_GUARD
                    eff_theta = min(0.92, base_theta + 0.25)

            elif control_strategy == "stat_baseline":
                active_prob_hvac = coarse_stat_prob
            else:
                active_prob_hvac = jev_prob_hvac

            # IVI と バッテリー温調の確率も算出
            active_prob_ivi = min(0.995, max(0.005, active_prob_hvac * 0.85 + stat_prob_ivi * 0.15))
            active_prob_batt = min(0.995, max(0.005, active_prob_hvac * 0.90 + stat_prob_batt * 0.10))

            # ROC評価用の正解ラベル (Ground Truth):
            # 「残り時間が HVAC リードタイム (12分) + 余裕窓 (4分) = 16分以内であるか」
            gt_should_wake_hvac = rem_min <= (self.specs.hvac_lead_time_minutes + 4.0)

            # 4. 制御戦略に応じた4機器の状態遷移決定
            if control_strategy == "always_on":
                state.hvac_state = HVACState.PRE_CONDITIONING if abs(state.cabin_temp_c - state.target_cabin_temp_c) > 1.0 else HVACState.READY_MAINTAIN
                state.ivi_state = IVIState.READY
                if state.ambient_temp_c <= self.specs.batt_cold_threshold_c or state.ambient_temp_c >= self.specs.batt_hot_threshold_c:
                    state.batt_thermal_state = BatteryThermalState.OPTIMAL_READY
                state.sentry_state = SentryState.FULL_AI_SENTRY
                pred_wake_hvac = True

            elif control_strategy == "no_precon":
                state.hvac_state = HVACState.OFF
                state.ivi_state = IVIState.DEEP_SLEEP
                state.batt_thermal_state = BatteryThermalState.OFF
                state.sentry_state = SentryState.LOW_POWER_PIR
                pred_wake_hvac = False

            else:
                # "psmp_jev", "psmp_lightgbm", または "stat_baseline"
                pred_wake_hvac = bool(active_prob_hvac >= eff_theta)

                # クールダウン中（一度予定超過でRe-sleepした直後）は無駄な再起動を抑制
                in_cooldown = t < re_sleep_cooldown_until

                # SoCが完全に帰還下限を割り込んだ場合（かつ途中充電非許容）は強制省電力カットオフ
                if soc_margin <= 0.3 and not profile.allow_enroute_charging:
                    soc_guard_triggered = True
                    current_policy = PowerPolicyMode.EMERGENCY_SAVE
                    state.hvac_state = HVACState.OFF
                    state.batt_thermal_state = BatteryThermalState.OFF
                    state.sentry_state = SentryState.LOW_POWER_PIR
                    if active_prob_ivi >= eff_theta:
                        state.ivi_state = IVIState.STANDBY
                else:
                    # --- ① 空調 (HVAC) の復帰・Re-sleep 制御 ---
                    cabin_gap = abs(state.cabin_temp_c - state.target_cabin_temp_c)
                    ambient_gap = abs(state.ambient_temp_c - state.target_cabin_temp_c)
                    need_rapid_precon = cabin_gap > 1.8
                    need_mild_maintain = cabin_gap > 0.8 or ambient_gap > 3.5

                    if state.hvac_active_minutes >= self.specs.hvac_max_hold_minutes:
                        # 予定時刻を大幅に過ぎてもドライバが戻らない場合 -> Re-sleep Guard 発動！
                        re_sleep_triggered = True
                        state.hvac_state = HVACState.ECO_HOLD if current_policy == PowerPolicyMode.COMFORT_FIRST else HVACState.OFF
                        state.hvac_active_minutes = 0.0
                        re_sleep_cooldown_until = t + 10.0
                    elif pred_wake_hvac and not in_cooldown:
                        if need_rapid_precon:
                            state.hvac_state = HVACState.PRE_CONDITIONING
                        elif need_mild_maintain and state.hvac_state == HVACState.OFF:
                            state.hvac_state = HVACState.READY_MAINTAIN
                    elif not pred_wake_hvac and state.hvac_state in (HVACState.PRE_CONDITIONING, HVACState.READY_MAINTAIN):
                        state.hvac_state = HVACState.ECO_HOLD if current_policy == PowerPolicyMode.COMFORT_FIRST else HVACState.OFF

                    # --- ② IVI / ナビ / 通信の復帰制御 ---
                    if state.ivi_active_minutes >= self.specs.ivi_max_hold_minutes and not pred_wake_hvac:
                        state.ivi_state = IVIState.STANDBY
                        state.ivi_active_minutes = 0.0
                    elif active_prob_ivi >= max(0.15, eff_theta - 0.05) or pred_wake_hvac:
                        if state.ivi_state == IVIState.DEEP_SLEEP:
                            state.ivi_state = IVIState.WARM_BOOT
                        elif state.ivi_state == IVIState.STANDBY:
                            state.ivi_state = IVIState.READY

                    # --- ③ 駆動用バッテリー温調の復帰制御 ---
                    need_batt_thermal = (
                        state.ambient_temp_c <= self.specs.batt_cold_threshold_c
                        or state.ambient_temp_c >= self.specs.batt_hot_threshold_c
                    )
                    if state.batt_active_minutes >= self.specs.batt_max_hold_minutes:
                        state.batt_thermal_state = BatteryThermalState.OFF
                        state.batt_active_minutes = 0.0
                    elif need_batt_thermal and active_prob_batt >= max(0.10, eff_theta - 0.04) and not in_cooldown:
                        if state.batt_thermal_state == BatteryThermalState.OFF:
                            state.batt_thermal_state = BatteryThermalState.PRE_HEAT_COOL

                    # --- ④ Sentry カメラの動的省電力調整 ---
                    if soc_margin < 5.0 and state.sentry_state == SentryState.FULL_AI_SENTRY:
                        state.sentry_state = SentryState.LOW_POWER_PIR

            # 5. 物理ステップ更新（温度・電力・SoC変化）
            prev_elapsed = state.elapsed_minutes
            state = self.sim.step_physics(state, dt_minutes=step_dt)

            timeline.append(
                TimeStepLog(
                    step_index=step_idx,
                    elapsed_minutes=round(prev_elapsed, 1),
                    remaining_minutes_to_departure=round(rem_min, 1),
                    stat_hazard_prob=coarse_stat_prob,
                    jev_wake_hvac_prob=round(jev_prob_hvac, 4),
                    lgbm_wake_hvac_prob=round(lgbm_prob_hvac, 4),
                    active_wake_hvac_prob=round(active_prob_hvac, 4),
                    jev_wake_ivi_prob=round(active_prob_ivi, 4),
                    jev_wake_batt_prob=round(active_prob_batt, 4),
                    effective_threshold_theta=round(eff_theta, 3),
                    ground_truth_should_wake_hvac=gt_should_wake_hvac,
                    predicted_wake_hvac=pred_wake_hvac,
                    hvac_state=state.hvac_state.value,
                    ivi_state=state.ivi_state.value,
                    batt_thermal_state=state.batt_thermal_state.value,
                    sentry_state=state.sentry_state.value,
                    policy_mode=current_policy.value,
                    cabin_temp_c=state.cabin_temp_c,
                    ambient_temp_c=state.ambient_temp_c,
                    battery_temp_c=state.battery_temp_c,
                    soc_pct=state.soc_pct,
                    step_power_w=state.step_power_w,
                    cumulative_energy_wh=state.cumulative_energy_wh,
                    re_sleep_triggered=re_sleep_triggered,
                    soc_guard_triggered=soc_guard_triggered,
                )
            )
            step_idx += 1

        # 6. ドライバ帰還時 (IG-ON) のレディ状態・快適度評価と帰宅エネルギー評価
        readiness = self.sim.evaluate_departure_readiness(state)
        est_home_arrival_soc = round(state.soc_pct - initial_plan.estimated_home_drive_soc_pct, 2)
        can_return_without_charge = est_home_arrival_soc >= max(4.0, profile.min_home_arrival_soc_pct * 0.65)
        enroute_charge_needed = not can_return_without_charge

        return ParkingSessionSimulationResult(
            session_id=event.session_id,
            driver_id=event.driver_id,
            poi_id=event.poi_id,
            poi_name_ja=initial_plan.poi_spec.name_ja,
            anomaly_type=event.anomaly_type,
            has_passengers=event.has_passengers,
            is_holiday=event.is_holiday,
            season=event.season,
            ambient_temp_c=event.ambient_temp_c,
            actual_duration_minutes=event.duration_minutes,
            expected_mean_minutes=initial_plan.stay_distribution.mean_minutes,
            control_strategy=control_strategy,
            roc_threshold_theta=round(base_theta, 3),
            initial_policy_mode=initial_plan.policy_mode.value,
            departure_cabin_temp_c=state.cabin_temp_c,
            target_cabin_temp_c=state.target_cabin_temp_c,
            departure_temp_error_c=readiness["temp_error_c"],
            cabin_comfort_score=readiness["cabin_comfort_score"],
            ivi_ready_score=readiness["ivi_ready_score"],
            batt_ready_score=readiness["batt_ready_score"],
            overall_ready_score=readiness["overall_ready_score"],
            is_ready_success=readiness["is_ready_success"],
            initial_soc_pct=event.initial_soc_pct,
            departure_soc_pct=state.soc_pct,
            parking_energy_consumed_wh=state.cumulative_energy_wh,
            estimated_home_arrival_soc_pct=est_home_arrival_soc,
            min_home_arrival_soc_pct=profile.min_home_arrival_soc_pct,
            can_return_home_without_charging=can_return_without_charge,
            enroute_charging_required=enroute_charge_needed,
            timeline_logs=timeline,
        )
