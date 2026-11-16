"""
Vehicle Devices Model (HVAC, IVI, Battery Thermal Management, Sentry Camera).
Models state transitions, power consumption (W / Wh), cabin & battery thermodynamics,
and readiness status upon driver return.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import math

from psmp.config import DEVICE_SPECS, DeviceHardwareSpec


class HVACState(str, Enum):
    OFF = "OFF"                       # 完全停止 (0W)
    ECO_HOLD = "ECO_HOLD"             # 弱換気・室温維持 (320W)
    PRE_CONDITIONING = "PRE_COND"     # 事前冷房 / 事前暖房 フル稼働 (2500~3200W)
    READY_MAINTAIN = "READY"          # 目標温度到達後の維持レディ (750W)


class IVIState(str, Enum):
    DEEP_SLEEP = "DEEP_SLEEP"         # ディープスリープ・RTC待機のみ (2.5W)
    STANDBY = "STANDBY"               # Suspend-to-RAM 高速復帰待機 (18W)
    WARM_BOOT = "WARM_BOOT"           # ナビ・通信・キャッシュ事前ロード中 (65W)
    READY = "READY"                   # 即時発進可能・画面即表示レディ (45W)


class BatteryThermalState(str, Enum):
    OFF = "OFF"                       # 温調停止 (0W)
    PRE_HEAT_COOL = "PRE_HEAT_COOL"   # 厳冬期プレヒート / 猛暑冷却 (2400W)
    OPTIMAL_READY = "OPTIMAL"         # 最適温度帯 (15~30℃) 到達・保温 (180W)


class SentryState(str, Enum):
    OFF = "OFF"                       # 監視オフ (0W)
    LOW_POWER_PIR = "LOW_POWER_PIR"   # 焦電/衝撃センサー待機＋イベント録画 (22W)
    FULL_AI_SENTRY = "FULL_AI_SENTRY" # 4カメラ常時AI推論監視 (240W)


class PowerPolicyMode(str, Enum):
    COMFORT_FIRST = "comfort_first"           # 快適性最優先（途中充電許容 or 余裕あり）
    BALANCED = "balanced"                     # 電費と快適性のバランス制御
    HOME_RANGE_GUARD = "home_range_guard"     # 自宅無充電帰還死守モード（省電力優先）
    EMERGENCY_SAVE = "emergency_save"         # SoC限界時の強制省電力カットオフ


@dataclass
class VehicleEnvironmentState:
    """
    各タイムステップ t における車両・車載4機器・熱環境・バッテリーの状態スナップショット
    """
    elapsed_minutes: float
    ambient_temp_c: float
    cabin_temp_c: float
    target_cabin_temp_c: float
    battery_temp_c: float
    soc_pct: float
    battery_energy_kwh: float
    battery_capacity_kwh: float

    # 4機器の現在状態
    hvac_state: HVACState = HVACState.OFF
    ivi_state: IVIState = IVIState.DEEP_SLEEP
    batt_thermal_state: BatteryThermalState = BatteryThermalState.OFF
    sentry_state: SentryState = SentryState.LOW_POWER_PIR

    # プレコン出力強度 (0.0 ~ 1.0, Jev Scoreで制御)
    precon_intensity: float = 0.8

    # 各機器がレディ/稼働状態の継続時間（遅延時のRe-sleep判定用）
    hvac_active_minutes: float = 0.0
    ivi_active_minutes: float = 0.0
    batt_active_minutes: float = 0.0

    # 累積消費電力量 (Wh)
    cumulative_energy_wh: float = 0.0
    step_power_w: float = 0.0


class VehicleSimulator:
    """
    車載4機器の電力消費および車内温度・バッテリー温度・SoCの時間発展シミュレータ
    """

    def __init__(self, specs: DeviceHardwareSpec = DEVICE_SPECS):
        self.specs = specs

    def compute_target_cabin_temp(self, ambient_temp_c: float) -> float:
        if ambient_temp_c >= 28.0:
            return self.specs.target_cabin_temp_summer_c
        elif ambient_temp_c <= 12.0:
            return self.specs.target_cabin_temp_winter_c
        return self.specs.target_cabin_temp_mild_c

    def compute_instant_power_w(
        self,
        hvac_state: HVACState,
        ivi_state: IVIState,
        batt_state: BatteryThermalState,
        sentry_state: SentryState,
        precon_intensity: float = 0.85,
    ) -> Dict[str, float]:
        # 1. HVAC Power
        intensity_scale = max(0.4, min(1.0, float(precon_intensity)))
        if hvac_state == HVACState.OFF:
            p_hvac = self.specs.hvac_power_off_w
        elif hvac_state == HVACState.ECO_HOLD:
            p_hvac = self.specs.hvac_power_eco_hold_w
        elif hvac_state == HVACState.PRE_CONDITIONING:
            p_hvac = self.specs.hvac_power_precon_max_w * intensity_scale
        else:  # READY_MAINTAIN
            p_hvac = self.specs.hvac_power_ready_maintain_w * (0.7 + 0.3 * intensity_scale)

        # 2. IVI Power
        if ivi_state == IVIState.DEEP_SLEEP:
            p_ivi = self.specs.ivi_power_deep_sleep_w
        elif ivi_state == IVIState.STANDBY:
            p_ivi = self.specs.ivi_power_standby_w
        elif ivi_state == IVIState.WARM_BOOT:
            p_ivi = self.specs.ivi_power_warm_boot_w
        else:  # READY
            p_ivi = self.specs.ivi_power_ready_w

        # 3. Battery Thermal Power
        if batt_state == BatteryThermalState.OFF:
            p_batt = self.specs.batt_power_off_w
        elif batt_state == BatteryThermalState.PRE_HEAT_COOL:
            p_batt = self.specs.batt_power_preheat_w * intensity_scale
        else:  # OPTIMAL_READY
            p_batt = self.specs.batt_power_optimal_hold_w

        # 4. Sentry Power
        if sentry_state == SentryState.OFF:
            p_sentry = self.specs.sentry_power_off_w
        elif sentry_state == SentryState.LOW_POWER_PIR:
            p_sentry = self.specs.sentry_power_low_pir_w
        else:  # FULL_AI_SENTRY
            p_sentry = self.specs.sentry_power_full_ai_w

        total_w = p_hvac + p_ivi + p_batt + p_sentry
        return {
            "hvac_w": p_hvac,
            "ivi_w": p_ivi,
            "batt_w": p_batt,
            "sentry_w": p_sentry,
            "total_w": total_w,
        }

    def step_physics(
        self,
        state: VehicleEnvironmentState,
        dt_minutes: float,
        solar_gain_c: float = 4.0,
    ) -> VehicleEnvironmentState:
        """
        dt_minutes 分間の時間経過による車内温度・バッテリー温度・消費電力・SoCを更新する
        """
        powers = self.compute_instant_power_w(
            state.hvac_state,
            state.ivi_state,
            state.batt_thermal_state,
            state.sentry_state,
            state.precon_intensity,
        )
        total_w = powers["total_w"]
        energy_wh = total_w * (dt_minutes / 60.0)

        # --- 車内温度 (Cabin Temperature) の熱力学モデル ---
        # 外気＋日射による平衡温度 (夏場は日射で外気温+8~12℃まで上昇、冬場は外気温へ漸近)
        if state.ambient_temp_c >= 26.0:
            equilibrium_temp = state.ambient_temp_c + solar_gain_c * 2.0
        elif state.ambient_temp_c <= 10.0:
            equilibrium_temp = state.ambient_temp_c
        else:
            equilibrium_temp = state.ambient_temp_c + solar_gain_c * 0.5

        cabin_c = state.cabin_temp_c
        target_c = state.target_cabin_temp_c
        intensity = max(0.4, min(1.0, state.precon_intensity))

        if state.hvac_state == HVACState.OFF:
            # 自然放熱・日射加熱 (時定数 τ ≈ 28分)
            alpha_natural = 1.0 - math.exp(-dt_minutes / 28.0)
            cabin_c = cabin_c + (equilibrium_temp - cabin_c) * alpha_natural
        elif state.hvac_state == HVACState.ECO_HOLD:
            # 弱維持 (時定数 τ ≈ 75分でゆっくりしか外気へ近づかない)
            alpha_eco = 1.0 - math.exp(-dt_minutes / 75.0)
            mid_target = 0.55 * target_c + 0.45 * equilibrium_temp
            cabin_c = cabin_c + (mid_target - cabin_c) * alpha_eco
        elif state.hvac_state == HVACState.PRE_CONDITIONING:
            # 急速冷暖房 (時定数 τ ≈ 6.5分 / intensity) -> 約10~12分で快適温度帯に到達
            tau_precon = 6.5 / intensity
            alpha_pre = 1.0 - math.exp(-dt_minutes / tau_precon)
            cabin_c = cabin_c + (target_c - cabin_c) * alpha_pre
            # 目標温度との差が 1.0℃ 以内になったら READY_MAINTAIN へ自動遷移可能
            if abs(cabin_c - target_c) <= 1.0:
                state.hvac_state = HVACState.READY_MAINTAIN
        elif state.hvac_state == HVACState.READY_MAINTAIN:
            # 目標温度付近でキープ
            alpha_ready = 1.0 - math.exp(-dt_minutes / 5.0)
            cabin_c = cabin_c + (target_c - cabin_c) * alpha_ready

        # --- 駆動用バッテリー温度 (Battery Temperature) モデル ---
        batt_c = state.battery_temp_c
        optimal_batt_c = 22.0
        if state.batt_thermal_state == BatteryThermalState.OFF:
            # バッテリーは熱容量が大きいため時定数 τ ≈ 90分 で外気温へ漸近
            alpha_b_nat = 1.0 - math.exp(-dt_minutes / 90.0)
            batt_c = batt_c + (state.ambient_temp_c - batt_c) * alpha_b_nat
        elif state.batt_thermal_state == BatteryThermalState.PRE_HEAT_COOL:
            # プレヒート / 冷却 (時定数 τ ≈ 10分)
            alpha_b_pre = 1.0 - math.exp(-dt_minutes / (10.0 / intensity))
            batt_c = batt_c + (optimal_batt_c - batt_c) * alpha_b_pre
            if 15.0 <= batt_c <= 29.0:
                state.batt_thermal_state = BatteryThermalState.OPTIMAL_READY
        elif state.batt_thermal_state == BatteryThermalState.OPTIMAL_READY:
            alpha_b_opt = 1.0 - math.exp(-dt_minutes / 12.0)
            batt_c = batt_c + (optimal_batt_c - batt_c) * alpha_b_opt

        # --- IVI 起動遷移 ---
        if state.ivi_state == IVIState.WARM_BOOT and state.ivi_active_minutes >= 2.0:
            state.ivi_state = IVIState.READY

        # --- 稼働継続時間のカウント更新 ---
        if state.hvac_state in (HVACState.PRE_CONDITIONING, HVACState.READY_MAINTAIN):
            state.hvac_active_minutes += dt_minutes
        else:
            state.hvac_active_minutes = 0.0

        if state.ivi_state in (IVIState.WARM_BOOT, IVIState.READY):
            state.ivi_active_minutes += dt_minutes
        else:
            state.ivi_active_minutes = 0.0

        if state.batt_thermal_state in (BatteryThermalState.PRE_HEAT_COOL, BatteryThermalState.OPTIMAL_READY):
            state.batt_active_minutes += dt_minutes
        else:
            state.batt_active_minutes = 0.0

        # --- バッテリー残量 (SoC) の更新 ---
        new_energy_kwh = max(0.0, state.battery_energy_kwh - (energy_wh / 1000.0))
        new_soc_pct = (new_energy_kwh / max(1e-6, state.battery_capacity_kwh)) * 100.0

        state.elapsed_minutes += dt_minutes
        state.cabin_temp_c = round(cabin_c, 2)
        state.battery_temp_c = round(batt_c, 2)
        state.battery_energy_kwh = round(new_energy_kwh, 4)
        state.soc_pct = round(new_soc_pct, 3)
        state.cumulative_energy_wh = round(state.cumulative_energy_wh + energy_wh, 2)
        state.step_power_w = round(total_w, 1)

        return state

    def evaluate_departure_readiness(self, state: VehicleEnvironmentState) -> Dict[str, float]:
        """
        ドライバが車に戻って来た瞬間 (IG-ON) における「快適度・レディ状態」を定量評価する
        """
        temp_error_c = abs(state.cabin_temp_c - state.target_cabin_temp_c)
        # 温度誤差 2.0℃ 以内なら快適度 85~100点、離れるほど低下
        cabin_comfort_score = max(0.0, min(100.0, 100.0 * math.exp(-((temp_error_c / 4.2) ** 2))))

        # IVIレディスコア (READY=100, WARM_BOOT=75, STANDBY=50, DEEP_SLEEP=15)
        ivi_scores = {
            IVIState.READY: 100.0,
            IVIState.WARM_BOOT: 75.0,
            IVIState.STANDBY: 50.0,
            IVIState.DEEP_SLEEP: 15.0,
        }
        ivi_ready_score = ivi_scores.get(state.ivi_state, 20.0)

        # 駆動バッテリー温調レディスコア (12℃〜32℃の間なら100点、氷点下や低温時は低下)
        if 12.0 <= state.battery_temp_c <= 32.0:
            batt_ready_score = 100.0
        elif state.battery_temp_c < 12.0:
            batt_ready_score = max(20.0, 100.0 - (12.0 - state.battery_temp_c) * 6.5)
        else:
            batt_ready_score = max(30.0, 100.0 - (state.battery_temp_c - 32.0) * 6.0)

        # 総合レディ・快適スコア (空調 60% + IVI 25% + バッテリー温調 15%)
        overall_ready_score = (
            0.60 * cabin_comfort_score
            + 0.25 * ivi_ready_score
            + 0.15 * batt_ready_score
        )
        is_ready_success = (temp_error_c <= 2.5) and (state.ivi_state in (IVIState.READY, IVIState.WARM_BOOT))

        return {
            "temp_error_c": round(temp_error_c, 2),
            "cabin_comfort_score": round(cabin_comfort_score, 1),
            "ivi_ready_score": round(ivi_ready_score, 1),
            "batt_ready_score": round(batt_ready_score, 1),
            "overall_ready_score": round(overall_ready_score, 1),
            "is_ready_success": bool(is_ready_success),
        }
