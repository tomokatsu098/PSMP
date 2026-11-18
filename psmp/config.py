"""
Configuration, constants, driver profiles, and POI definitions for PSMP.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class POIType( str, Enum ):
    HOME = "home"                             # 自宅 (長大時間: 8~14h または 24h超長期)
    CONVENIENCE_STORE = "convenience_store"   # コンビニ (短時間: 10~15分)
    SUPERMARKET = "supermarket"               # スーパー・買い物 (中時間: 35~60分)
    RESTAURANT_LEISURE = "restaurant_leisure" # レストラン・映画・レジャー (中長時間: 70~140分)
    OFFICE_WORK = "office_work"               # オフィス・出勤先 (長時間: 240~540分)
    GYM_CLINIC = "gym_clinic"                 # ジム・病院・習い事 (中時間: 60~90分)


class PowertrainType(str, Enum):
    """
    車両パワートレイン種別:
      - BEV: 純電気自動車 (40~100kWh 大容量高圧バッテリ + 12V補機。高圧SoCによる12V自動充電連動)
      - PHEV: プラグインハイブリッド (10~20kWh 中容量高圧バッテリ + 12V補機 + ICE。ガソリン自走可能だが12V枯渇防止)
      - HEV: ストロングハイブリッド (1~2kWh 小容量高圧バッテリ + 12V補機 + ICE。高圧プレ空調不可・12V暗電流最小化が命)
      - ICE: 内燃機関車 (ガソリン/ディーゼル、12V鉛バッテリのみ。セル始動性死守・プレ空調なし・DeepSleep最大化)
    """
    BEV = "bev"
    PHEV = "phev"
    HEV = "hev"
    ICE = "ice"


def classify_stay_duration_category(duration_minutes: float, is_home: bool = False) -> str:
    """
    駐車滞在時間を自宅/出先および時間幅に応じて分類する:
      - 自宅 (is_home=True):
        - HOME_OVERNIGHT: <= 24 時間 (1440分) - 日常の帰宅・夜間駐車
        - HOME_LONG: > 24 時間 (1440分) - 週末放置や数日以上の長期駐車 (委託元の3日以上長期ターゲット含む)
      - 出先 (is_home=False):
        - QUICK: <= 25 分 (コンビニ、給油、送迎など)
        - MEDIUM: <= 90 分 (スーパー、買い物、外食など)
        - LONG: > 90 分 (オフィス、レジャー、病院、半日用事など)
    """
    if is_home:
        if duration_minutes <= 1440.0:  # 24時間以内
            return "HOME_OVERNIGHT"
        return "HOME_LONG"

    if duration_minutes <= 25.0:
        return "QUICK"
    elif duration_minutes <= 90.0:
        return "MEDIUM"
    return "LONG"



@dataclass(frozen=True)
class POIClusterSpec:
    poi_id: str
    name_ja: str
    poi_type: POIType
    lat: float
    lon: float
    distance_to_home_km: float
    security_risk_level: float  # 0.0 (安全) ~ 1.0 (高リスク: Sentry優先度高)
    base_mean_minutes: float
    base_std_minutes: float


def get_poi_default_category(poi: POIClusterSpec) -> str:
    """
    POIスペックの種別や滞在平均時間からデフォルトの滞在カテゴリを取得する
    """
    if poi.poi_type == POIType.HOME or poi.distance_to_home_km <= 0.05:
        return "HOME_OVERNIGHT"
    elif poi.poi_type == POIType.CONVENIENCE_STORE:
        return "QUICK"
    elif poi.poi_type in (POIType.SUPERMARKET, POIType.GYM_CLINIC):
        return "MEDIUM"
    elif poi.poi_type in (POIType.RESTAURANT_LEISURE, POIType.OFFICE_WORK):
        return "LONG" if poi.base_mean_minutes > 90.0 else "MEDIUM"
    return classify_stay_duration_category(poi.base_mean_minutes, is_home=(poi.poi_type == POIType.HOME))



DEFAULT_POI_CLUSTERS: Dict[str, POIClusterSpec] = {
    "POI_HOME_00": POIClusterSpec(
        poi_id="POI_HOME_00",
        name_ja="自宅ガレージ・専用駐車場",
        poi_type=POIType.HOME,
        lat=35.6800,
        lon=139.7600,
        distance_to_home_km=0.0,
        security_risk_level=0.05,
        base_mean_minutes=720.0,
        base_std_minutes=120.0,
    ),
    "POI_CONV_01": POIClusterSpec(
        poi_id="POI_CONV_01",
        name_ja="セブンイレブン（幹線道路沿い）",
        poi_type=POIType.CONVENIENCE_STORE,
        lat=35.6812,
        lon=139.7671,
        distance_to_home_km=12.0,
        security_risk_level=0.35,
        base_mean_minutes=12.0,
        base_std_minutes=4.0,
    ),
    "POI_SUPER_02": POIClusterSpec(
        poi_id="POI_SUPER_02",
        name_ja="大型スーパー・ショッピングモール",
        poi_type=POIType.SUPERMARKET,
        lat=35.6580,
        lon=139.7515,
        distance_to_home_km=18.5,
        security_risk_level=0.55,
        base_mean_minutes=50.0,
        base_std_minutes=12.0,
    ),
    "POI_REST_03": POIClusterSpec(
        poi_id="POI_REST_03",
        name_ja="郊外ファミリーレストラン＆シネマ",
        poi_type=POIType.RESTAURANT_LEISURE,
        lat=35.6295,
        lon=139.7745,
        distance_to_home_km=32.0,
        security_risk_level=0.65,
        base_mean_minutes=95.0,
        base_std_minutes=20.0,
    ),
    "POI_OFFICE_04": POIClusterSpec(
        poi_id="POI_OFFICE_04",
        name_ja="都心オフィスビル駐車場",
        poi_type=POIType.OFFICE_WORK,
        lat=35.6895,
        lon=139.6917,
        distance_to_home_km=25.0,
        security_risk_level=0.25,
        base_mean_minutes=180.0,  # シミュレーション評価用に3時間枠を代表値に設定
        base_std_minutes=25.0,
    ),
    "POI_GYM_05": POIClusterSpec(
        poi_id="POI_GYM_05",
        name_ja="スポーツジム・クリニックモール",
        poi_type=POIType.GYM_CLINIC,
        lat=35.6721,
        lon=139.7234,
        distance_to_home_km=15.0,
        security_risk_level=0.40,
        base_mean_minutes=70.0,
        base_std_minutes=10.0,
    ),
}


@dataclass
class DriverPreferenceProfile:
    """
    ドライバごとの個人嗜好と車両パワートレイン・バッテリーマネジメントのトレードオフ設定
    """
    driver_id: str
    name_ja: str
    description_ja: str
    # パワートレイン種別 (BEV, PHEV, HEV, ICE)
    powertrain_type: PowertrainType = PowertrainType.BEV
    # 0.0 = 途中充電やむなし・快適性最優先, 1.0 = 絶対に自宅まで無充電で帰りたい（電費最優先）
    home_charge_priority_alpha: float = 0.50
    # 0.0 = 室温にこだわらない, 1.0 = 夏場の冷房・冬場の暖房による入車時快適性を最優先
    comfort_priority_beta: float = 0.50
    # 途中充電を許容するか (True: 快適性のためなら途中経路での急速充電を許容)
    allow_enroute_charging: bool = False
    # ROC感度特性に基づく復帰判定閾値 θ (0.0 ~ 1.0)
    # 低いほど「早めに・高感度で機器をレディ復帰」(TPR高・FPR高)、高いほど「確実な時のみ復帰」(省電力・FPR低)
    roc_threshold_theta: float = 0.45
    # 自宅帰還時に最低限残しておきたいバッテリー安全マージン SoC (%)
    min_home_arrival_soc_pct: float = 12.0
    # 車両バッテリー総容量 (kWh) - 例: EV 60kWh / PHEV 13.6kWh / HEV 1.8kWh / ICE 0.72kWh(12V 60Ah)
    battery_capacity_kwh: float = 60.0
    # 平均電費 (km / kWh) または 相当燃費
    efficiency_km_per_kwh: float = 6.5
    # 12V補機バッテリー容量 (Ah)
    auxiliary_12v_capacity_ah: float = 60.0
    # 高圧バッテリーから12V補機への駐車中自動DC-DC充電に対応しているか
    supports_auto_dcdc_charging: bool = True


DEFAULT_DRIVER_PROFILES: Dict[str, DriverPreferenceProfile] = {
    "DRV_A_ECO_HOME": DriverPreferenceProfile(
        driver_id="DRV_A_ECO_HOME",
        name_ja="ドライバA（自宅充電こだわり・電費優先派・BEV）",
        description_ja="自宅ソーラー・夜間電力で充電するため、出先での途中充電を嫌い、無充電帰宅を最優先するBEVドライバ。",
        powertrain_type=PowertrainType.BEV,
        home_charge_priority_alpha=0.85,
        comfort_priority_beta=0.35,
        allow_enroute_charging=False,
        roc_threshold_theta=0.60,
        min_home_arrival_soc_pct=15.0,
        battery_capacity_kwh=60.0,
        efficiency_km_per_kwh=6.8,
        supports_auto_dcdc_charging=True,
    ),
    "DRV_B_COMFORT_FIRST": DriverPreferenceProfile(
        driver_id="DRV_B_COMFORT_FIRST",
        name_ja="ドライバB（車内快適性最優先・途中充電許容派・BEV）",
        description_ja="夏場の冷房・冬場の暖房で快適に入車することを最優先し、必要なら帰路での継ぎ足し充電も厭わないBEVドライバ。",
        powertrain_type=PowertrainType.BEV,
        home_charge_priority_alpha=0.20,
        comfort_priority_beta=0.90,
        allow_enroute_charging=True,
        roc_threshold_theta=0.30,
        min_home_arrival_soc_pct=8.0,
        battery_capacity_kwh=60.0,
        efficiency_km_per_kwh=6.2,
        supports_auto_dcdc_charging=True,
    ),
    "DRV_C_PHEV_BALANCED": DriverPreferenceProfile(
        driver_id="DRV_C_PHEV_BALANCED",
        name_ja="ドライバC（プリウスPHEV想定・バランス＆同乗者配慮派・PHEV）",
        description_ja="普段はEV走行で自宅帰還を目指すが、ガソリン自走可能なため電欠耐性が高く、猛暑・厳冬時は快適性を優先する。",
        powertrain_type=PowertrainType.PHEV,
        home_charge_priority_alpha=0.55,
        comfort_priority_beta=0.65,
        allow_enroute_charging=False,
        roc_threshold_theta=0.45,
        min_home_arrival_soc_pct=12.0,
        battery_capacity_kwh=15.0,  # PHEV容量
        efficiency_km_per_kwh=7.2,
        supports_auto_dcdc_charging=True,
    ),
    "DRV_D_HEV_STANDARD": DriverPreferenceProfile(
        driver_id="DRV_D_HEV_STANDARD",
        name_ja="ドライバD（アクア/プリウスHEV想定・12V暗電流死守・HEV）",
        description_ja="1.8kWh小型高圧バッテリー＋12V鉛補機。高圧プレ空調は非装備、12V暗電流カットによる30日放置上がり防止を最優先。",
        powertrain_type=PowertrainType.HEV,
        home_charge_priority_alpha=0.50,
        comfort_priority_beta=0.30,
        allow_enroute_charging=False,
        roc_threshold_theta=0.50,
        min_home_arrival_soc_pct=20.0,
        battery_capacity_kwh=1.8,   # HEV容量
        efficiency_km_per_kwh=22.0, # km/L相当
        supports_auto_dcdc_charging=False,
    ),
    "DRV_E_ICE_COMMUTER": DriverPreferenceProfile(
        driver_id="DRV_E_ICE_COMMUTER",
        name_ja="ドライバE（ガソリン車・クランキング始動性最優先・ICE）",
        description_ja="12V鉛バッテリーのみ搭載。セルモーター起動電流（数十〜数百A）を死守するため、駐車中はDeepSleep最大化を徹底。",
        powertrain_type=PowertrainType.ICE,
        home_charge_priority_alpha=0.50,
        comfort_priority_beta=0.20,
        allow_enroute_charging=False,
        roc_threshold_theta=0.55,
        min_home_arrival_soc_pct=30.0,
        battery_capacity_kwh=0.72,  # 12V 60Ah = 約0.72kWh
        efficiency_km_per_kwh=16.0, # km/L相当
        supports_auto_dcdc_charging=False,
    ),
}


@dataclass(frozen=True)
class DeviceHardwareSpec:
    """
    車載機器 1〜4 の電力・リードタイム仕様
    """
    # ① 空調 (HVAC: 冷暖房プレコンディショニング)
    hvac_power_off_w: float = 0.0
    hvac_power_eco_hold_w: float = 320.0
    hvac_power_precon_max_w: float = 3200.0
    hvac_power_ready_maintain_w: float = 750.0
    hvac_lead_time_minutes: float = 12.0
    hvac_max_hold_minutes: float = 20.0  # 予定遅延時に再スリープするまでの最大保持時間

    # ② 車載インフォテインメント (IVI) / ナビ / 通信モジュール
    ivi_power_deep_sleep_w: float = 2.5
    ivi_power_standby_w: float = 18.0
    ivi_power_warm_boot_w: float = 65.0
    ivi_power_ready_w: float = 45.0
    ivi_lead_time_minutes: float = 3.0
    ivi_max_hold_minutes: float = 15.0

    # ③ 駆動用バッテリー温調 (Battery Thermal Management)
    batt_power_off_w: float = 0.0
    batt_power_preheat_w: float = 2400.0
    batt_power_optimal_hold_w: float = 180.0
    batt_lead_time_minutes: float = 16.0
    batt_max_hold_minutes: float = 22.0
    batt_cold_threshold_c: float = 8.0   # 外気温8℃以下でプレヒート対象
    batt_hot_threshold_c: float = 35.0   # 外気温35℃以上でバッテリー冷却対象

    # ④ 監視カメラ / セキュリティシステム (Sentry Mode)
    sentry_power_off_w: float = 0.0
    sentry_power_low_pir_w: float = 22.0
    sentry_power_full_ai_w: float = 240.0

    # 目標快適車内温度 (℃)
    target_cabin_temp_summer_c: float = 23.5
    target_cabin_temp_winter_c: float = 22.5
    target_cabin_temp_mild_c: float = 23.0


DEVICE_SPECS = DeviceHardwareSpec()
