"""
Phase 1: Data Collector & Learner (`phase1_collector_learner.py`)
１．学習すべき情報を収集するフェーズ
  - ドライバIDごとの過去に駐車したタイミング (IG-OFF) と、駐車から走り出したタイミング (IG-ON)
  - 座標 (緯度・経度・POIクラスタ)
  - 日時 (日付、時間、曜日、平日/休日、季節・外気温)
  - 同乗者有無 (has_passengers)
  - ドライバごとのバッテリーマネジメント個人嗜好 (自宅無充電帰還 vs 車内快適性優先度)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from psmp.config import (
    DEFAULT_DRIVER_PROFILES,
    DEFAULT_POI_CLUSTERS,
    DriverPreferenceProfile,
    POIClusterSpec,
    classify_stay_duration_category,
    get_poi_default_category,
)



@dataclass
class ParkingEventRecord:
    """
    1回の駐車セッション（IG-OFF から IG-ON まで）の記録データ構造
    """
    session_id: str
    driver_id: str
    poi_id: str
    lat: float
    lon: float
    park_start_dt: datetime        # 駐車したタイミング (IG-OFF)
    departure_dt: datetime         # 駐車から走り出したタイミング (IG-ON)
    duration_minutes: float        # 実滞在時間 (分)
    day_of_week: int               # 0=月曜 ~ 6=日曜
    is_holiday: bool               # 休日・祝日フラグ
    hour_of_day: int               # 駐車開始時間帯 (0~23)
    season: str                    # "summer" | "winter" | "spring_autumn"
    ambient_temp_c: float          # 外気温 (℃)
    has_passengers: bool           # 同乗者有無
    initial_soc_pct: float         # 駐車開始時のバッテリーSoC (%)
    distance_to_home_km: float     # 自宅までの距離 (km)
    anomaly_type: str = "normal"   # "normal" | "early_return" | "unexpected_delay" | "extreme_weather"


@dataclass
class LearnedStayDistribution:
    """
    Phase 1 がコンテキスト (DriverID × 座標POI × 平日/休日 × 時間帯 × 同乗者有無) ごとに
    学習した滞在時間確率分布パラメータ
    """
    mean_minutes: float
    std_minutes: float
    p10_minutes: float
    p50_minutes: float
    p90_minutes: float
    sample_count: int
    confidence_weight: float  # サンプル数に応じた統計信頼度 (0.0 ~ 1.0)
    raw_durations: Optional[np.ndarray] = None  # ECDF（経験的累積分布）用のソート済み実滞在時間データ


class Phase1CollectorLearner:
    """
    【Phase 1】学習すべき情報を収集し、ドライバ×座標×日時×同乗者ごとの滞在時間予測分布と
    ドライバ嗜好プロファイルを構築・更新するモジュール
    """

    def __init__(
        self,
        poi_clusters: Optional[Dict[str, POIClusterSpec]] = None,
        driver_profiles: Optional[Dict[str, DriverPreferenceProfile]] = None,
    ):
        self.poi_clusters = poi_clusters or dict(DEFAULT_POI_CLUSTERS)
        self.driver_profiles = {
            k: DriverPreferenceProfile(**vars(v))
            for k, v in (driver_profiles or DEFAULT_DRIVER_PROFILES).items()
        }
        self.history_records: List[ParkingEventRecord] = []
        self.learned_stats: Dict[Tuple[Any, ...], LearnedStayDistribution] = {}
        self.fallback_poi_stats: Dict[Tuple[str, str], LearnedStayDistribution] = {}
        # Cold-start（未学習POI/初訪問先）向けドライバ別カテゴリECDF & グローバル滞在時間
        self.driver_category_durations: Dict[Tuple[str, str], np.ndarray] = {}
        self.driver_global_durations: Dict[str, np.ndarray] = {}


    def resolve_nearest_poi(self, lat: float, lon: float) -> POIClusterSpec:
        """
        GPS座標 (lat, lon) から最も近い登録POIクラスタを特定する
        """
        best_poi = None
        best_dist = float("inf")
        for poi in self.poi_clusters.values():
            d = math.hypot(lat - poi.lat, lon - poi.lon)
            if d < best_dist:
                best_dist = d
                best_poi = poi
        assert best_poi is not None
        return best_poi

    @staticmethod
    def time_bucket(hour: int) -> str:
        if 6 <= hour < 11:
            return "morning"
        elif 11 <= hour < 16:
            return "daytime"
        elif 16 <= hour < 21:
            return "evening"
        return "night"

    def generate_synthetic_history(
        self,
        num_sessions_per_driver: int = 120,
        seed: int = 42,
        include_edge_cases: bool = True,
    ) -> List[ParkingEventRecord]:
        """
        ドライバIDごとのリアルな過去駐車履歴データ（トレーニング用＆評価テスト用）を生成する。
        - ドライバ・POI・平日/休日・時間帯・同乗者の有無による滞在時間の違いを反映
        - 評価用には「急な予定変更（早期帰還・大幅遅延）」「猛暑/厳冬」「低SoCギリギリ帰還」も含める
        """
        rng = np.random.default_rng(seed)
        records: List[ParkingEventRecord] = []
        base_date = datetime(2026, 6, 1, 8, 0, 0)
        poi_list = list(self.poi_clusters.values())

        session_idx = 0
        for driver_id, profile in self.driver_profiles.items():
            for i in range(num_sessions_per_driver):
                session_idx += 1
                day_offset = int(rng.integers(0, 180))
                day_dt = base_date + timedelta(days=day_offset)
                dow = day_dt.weekday()
                is_holiday = dow >= 5 or (day_offset % 29 == 0)

                # 平日と休日で訪問しやすいPOIと同乗者確率を変える
                if not is_holiday:
                    poi_probs = [0.25, 0.25, 0.10, 0.25, 0.15]  # 平日はコンビニ・スーパー・オフィス多め
                    has_passengers = bool(rng.random() < (0.45 if "COMFORT" in driver_id else 0.20))
                else:
                    poi_probs = [0.15, 0.35, 0.30, 0.05, 0.15]  # 休日はスーパー・レストラン・レジャー多め
                    has_passengers = bool(rng.random() < (0.75 if "COMFORT" in driver_id else 0.50))

                poi: POIClusterSpec = rng.choice(poi_list, p=poi_probs)  # type: ignore

                # 時間帯のサンプリング
                if poi.poi_id == "POI_OFFICE_04":
                    hour = int(rng.choice([8, 9, 10, 13]))
                elif poi.poi_id == "POI_REST_03":
                    hour = int(rng.choice([12, 13, 18, 19]))
                else:
                    hour = int(rng.integers(8, 21))
                minute = int(rng.integers(0, 60))
                park_start_dt = day_dt.replace(hour=hour, minute=minute)

                # ドライバ×POI×休日×同乗者に応じた真の期待滞在時間
                true_mean, true_std = self._true_stay_params(driver_id, poi, is_holiday, hour, has_passengers)

                # エッジケース（時に急な予定変更があるケース）の注入
                anomaly_type = "normal"
                if include_edge_cases and i >= int(num_sessions_per_driver * 0.75):
                    r_anom = rng.random()
                    if r_anom < 0.12:
                        # 予測よりかなり早く戻ってくるケース
                        anomaly_type = "early_return"
                        duration = max(6.0, true_mean * float(rng.uniform(0.45, 0.65)))
                    elif r_anom < 0.24:
                        # 話し込み・レジ混雑等で予定より大幅に遅れるケース
                        anomaly_type = "unexpected_delay"
                        duration = true_mean + max(22.0, true_mean * float(rng.uniform(0.40, 0.75)))
                    else:
                        duration = float(rng.normal(true_mean, true_std))
                else:
                    duration = float(rng.normal(true_mean, true_std))

                duration = round(max(6.0, min(240.0, duration)), 1)
                departure_dt = park_start_dt + timedelta(minutes=duration)

                # 季節と外気温の決定 (夏場: 31~38℃ / 冬場: -2~7℃ / 中間期: 16~24℃)
                season_pick = rng.choice(["summer", "winter", "spring_autumn"], p=[0.42, 0.38, 0.20])
                if season_pick == "summer":
                    ambient_c = round(float(rng.uniform(30.0, 38.5)), 1)
                elif season_pick == "winter":
                    ambient_c = round(float(rng.uniform(-3.0, 7.5)), 1)
                else:
                    ambient_c = round(float(rng.uniform(16.0, 24.5)), 1)

                # 駐車開始時のバッテリーSoC (%)
                # 帰宅必要SoCギリギリのシナリオも約20%混ぜてトレードオフ評価を可能にする
                req_home_kwh = (poi.distance_to_home_km / profile.efficiency_km_per_kwh)
                req_home_soc = (req_home_kwh / profile.battery_capacity_kwh) * 100.0 + profile.min_home_arrival_soc_pct
                if rng.random() < 0.22:
                    # 自宅無充電帰還に対して余裕が少ない（トレードオフが顕在化する）SoC状態
                    initial_soc = round(float(rng.uniform(req_home_soc + 1.5, req_home_soc + 7.5)), 2)
                else:
                    initial_soc = round(float(rng.uniform(max(35.0, req_home_soc + 10.0), 88.0)), 2)
                initial_soc = min(95.0, max(15.0, initial_soc))

                # GPS座標に微小な駐車枠ノイズを付加
                lat_noisy = round(poi.lat + float(rng.normal(0, 0.0004)), 6)
                lon_noisy = round(poi.lon + float(rng.normal(0, 0.0004)), 6)

                records.append(
                    ParkingEventRecord(
                        session_id=f"SES_{session_idx:04d}",
                        driver_id=driver_id,
                        poi_id=poi.poi_id,
                        lat=lat_noisy,
                        lon=lon_noisy,
                        park_start_dt=park_start_dt,
                        departure_dt=departure_dt,
                        duration_minutes=duration,
                        day_of_week=dow,
                        is_holiday=is_holiday,
                        hour_of_day=hour,
                        season=str(season_pick),
                        ambient_temp_c=ambient_c,
                        has_passengers=has_passengers,
                        initial_soc_pct=initial_soc,
                        distance_to_home_km=poi.distance_to_home_km,
                        anomaly_type=anomaly_type,
                    )
                )

        return records

    def _true_stay_params(
        self,
        driver_id: str,
        poi: POIClusterSpec,
        is_holiday: bool,
        hour: int,
        has_passengers: bool,
    ) -> Tuple[float, float]:
        """
        ドライバ個性×POI×日時×同乗者による潜在的な滞在時間特性
        """
        mean = poi.base_mean_minutes
        std = poi.base_std_minutes

        # ドライバ個性補正
        if "ECO_HOME" in driver_id:
            mean *= 0.90  # 用事を素早く済ませる傾向
            std *= 0.75
        elif "COMFORT_FIRST" in driver_id:
            mean *= 1.12  # ゆったり滞在する傾向
            std *= 0.85
        else:
            mean *= 1.00
            std *= 0.90

        # 休日は買い物・レジャーの滞在時間が伸びる
        if is_holiday and poi.poi_id in ("POI_SUPER_02", "POI_REST_03"):
            mean *= 1.22

        # 同乗者（家族・子供等）がいる場合は滞在時間が15〜20%長くなる
        if has_passengers:
            mean *= 1.18
            std *= 1.05

        # 夕方〜夜の買い物はやや短め
        if self.time_bucket(hour) == "evening" and poi.poi_id == "POI_SUPER_02":
            mean *= 0.88

        return round(mean, 2), round(max(2.5, std), 2)

    def fit(self, records: List[ParkingEventRecord]) -> None:
        """
        収集した駐車履歴データから、条件別の滞在時間分布（中央値・ロバスト分散・分位点・ECDFデータ）を学習する。
        さらに、未訪問先（Cold-start）向けにドライバごとの滞在時間カテゴリ別ECDF（個人癖×場所種別）および
        全出先滞在時間を構築・保持する。
        """
        self.history_records = list(records)
        grouped: Dict[Tuple[Any, ...], List[float]] = {}
        fallback_grouped: Dict[Tuple[str, str], List[float]] = {}
        driver_cat_map: Dict[Tuple[str, str], List[float]] = {}
        driver_all_map: Dict[str, List[float]] = {}

        for rec in records:
            tb = self.time_bucket(rec.hour_of_day)
            key = (rec.driver_id, rec.poi_id, rec.is_holiday, tb, rec.has_passengers)
            grouped.setdefault(key, []).append(rec.duration_minutes)

            fb_key = (rec.driver_id, rec.poi_id)
            fallback_grouped.setdefault(fb_key, []).append(rec.duration_minutes)

            # Cold-start 向けドライバ別カテゴリ（QUICK/MEDIUM/LONG）および全出先滞在時間
            cat = classify_stay_duration_category(rec.duration_minutes)
            driver_cat_map.setdefault((rec.driver_id, cat), []).append(rec.duration_minutes)
            driver_all_map.setdefault(rec.driver_id, []).append(rec.duration_minutes)

        self.learned_stats.clear()
        for key, durations in grouped.items():
            arr = np.sort(np.array(durations, dtype=np.float64))
            p50 = float(np.percentile(arr, 50))
            p25 = float(np.percentile(arr, 25))
            p75 = float(np.percentile(arr, 75))
            iqr = max(5.0, (p75 - p25) / 1.349)
            self.learned_stats[key] = LearnedStayDistribution(
                mean_minutes=round(p50, 2),  # 外れ値に強い中央値を代表値とする
                std_minutes=round(iqr, 2),   # 四分位範囲に基づくロバスト標準偏差
                p10_minutes=round(float(np.percentile(arr, 10)), 2),
                p50_minutes=round(p50, 2),
                p90_minutes=round(float(np.percentile(arr, 90)), 2),
                sample_count=len(arr),
                confidence_weight=round(min(1.0, len(arr) / 8.0), 3),
                raw_durations=arr,
            )

        self.fallback_poi_stats.clear()
        for fb_key, durations in fallback_grouped.items():
            arr = np.sort(np.array(durations, dtype=np.float64))
            p50 = float(np.percentile(arr, 50))
            p25 = float(np.percentile(arr, 25))
            p75 = float(np.percentile(arr, 75))
            iqr = max(6.0, (p75 - p25) / 1.349)
            self.fallback_poi_stats[fb_key] = LearnedStayDistribution(
                mean_minutes=round(p50, 2),
                std_minutes=round(iqr, 2),
                p10_minutes=round(float(np.percentile(arr, 10)), 2),
                p50_minutes=round(p50, 2),
                p90_minutes=round(float(np.percentile(arr, 90)), 2),
                sample_count=len(arr),
                confidence_weight=round(min(1.0, len(arr) / 12.0), 3),
                raw_durations=arr,
            )

        self.driver_category_durations.clear()
        for cat_key, cat_durations in driver_cat_map.items():
            self.driver_category_durations[cat_key] = np.sort(np.array(cat_durations, dtype=np.float64))

        self.driver_global_durations.clear()
        for drv_id, all_durations in driver_all_map.items():
            self.driver_global_durations[drv_id] = np.sort(np.array(all_durations, dtype=np.float64))

    def predict_stay_distribution(
        self,
        driver_id: str,
        lat: float,
        lon: float,
        park_start_dt: datetime,
        is_holiday: bool,
        has_passengers: bool,
        target_category: Optional[str] = None,
    ) -> Tuple[POIClusterSpec, LearnedStayDistribution]:
        """
        新しい駐車イベント発生時に、座標・日時・同乗者有無から滞在時間の予測分布を返す。
        - 既知のPOIクラスタで十分な履歴がある場合は、ベイズ縮小推定による条件別ECDF分布を使用。
        - 未学習POI（Cold-start）またはサンプル数不足の場合は、目的地の種別/ナビジャンルに応じた
          「ドライバ別カテゴリECDF（個人癖×場所種別）」を自動適用する。
        """
        poi = self.resolve_nearest_poi(lat, lon)
        tb = self.time_bucket(park_start_dt.hour)
        key = (driver_id, poi.poi_id, is_holiday, tb, has_passengers)

        exact_dist = self.learned_stats.get(key)
        fb_dist = self.fallback_poi_stats.get((driver_id, poi.poi_id))

        if exact_dist is not None and fb_dist is not None:
            # ベイズ縮小推定 (Empirical Bayes Shrinkage): サンプル数が少ない場合はPOI全体統計とブレンド
            w = exact_dist.confidence_weight
            mean_b = w * exact_dist.mean_minutes + (1.0 - w) * fb_dist.mean_minutes
            std_b = w * exact_dist.std_minutes + (1.0 - w) * fb_dist.std_minutes
            # 生データは条件キー側を優先（無ければフォールバック）
            raw_durs = exact_dist.raw_durations if (exact_dist.raw_durations is not None and len(exact_dist.raw_durations) >= 3) else fb_dist.raw_durations
            return poi, LearnedStayDistribution(
                mean_minutes=round(mean_b, 2),
                std_minutes=round(std_b, 2),
                p10_minutes=round(mean_b - 1.28 * std_b, 2),
                p50_minutes=round(mean_b, 2),
                p90_minutes=round(mean_b + 1.28 * std_b, 2),
                sample_count=exact_dist.sample_count,
                confidence_weight=exact_dist.confidence_weight,
                raw_durations=raw_durs,
            )
        elif fb_dist is not None and fb_dist.sample_count >= 3:
            return poi, fb_dist

        # Cold-start（未学習POI / 初訪問先）: ドライバ別カテゴリECDFを適用
        cat = target_category or get_poi_default_category(poi)
        cat_durs = self.driver_category_durations.get((driver_id, cat))
        if cat_durs is None or len(cat_durs) < 2:
            cat_durs = self.driver_global_durations.get(driver_id)

        if cat_durs is not None and len(cat_durs) >= 2:
            p50 = float(np.percentile(cat_durs, 50))
            p25 = float(np.percentile(cat_durs, 25))
            p75 = float(np.percentile(cat_durs, 75))
            iqr = max(5.0, (p75 - p25) / 1.349)
            return poi, LearnedStayDistribution(
                mean_minutes=round(p50, 2),
                std_minutes=round(iqr, 2),
                p10_minutes=round(float(np.percentile(cat_durs, 10)), 2),
                p50_minutes=round(p50, 2),
                p90_minutes=round(float(np.percentile(cat_durs, 90)), 2),
                sample_count=len(cat_durs),
                confidence_weight=0.80,
                raw_durations=cat_durs,
            )

        # 履歴が一切ない初期状態の場合はPOI基本スペックを返す
        return poi, LearnedStayDistribution(
            mean_minutes=poi.base_mean_minutes,
            std_minutes=poi.base_std_minutes,
            p10_minutes=round(poi.base_mean_minutes - 1.28 * poi.base_std_minutes, 2),
            p50_minutes=poi.base_mean_minutes,
            p90_minutes=round(poi.base_mean_minutes + 1.28 * poi.base_std_minutes, 2),
            sample_count=0,
            confidence_weight=0.2,
            raw_durations=None,
        )

    def compute_conditional_return_probability(
        self,
        dist: LearnedStayDistribution,
        elapsed_minutes: float,
        horizon_minutes: float,
    ) -> float:
        """
        現在すでに `elapsed_minutes` 分駐車しているという条件のもとで、
        今後 `horizon_minutes` 分以内（[t, t + Δt]）にドライバが戻ってくる条件付き確率（ハザード積分量）を計算する。
        - 経験的滞在データ (raw_durations) が存在する場合は ECDF（経験的累積分布 / カプラン・マイヤー生存関数）による直接ハザード計算を行う
        - 未学習POIなどの場合は従来の正規分布フォールバック
        """
        raw_durs = getattr(dist, "raw_durations", None)
        if raw_durs is not None and len(raw_durs) >= 3:
            surv_idx = np.searchsorted(raw_durs, elapsed_minutes, side="right")
            n_surv = len(raw_durs) - surv_idx
            if n_surv <= 0:
                return 0.005
            ret_idx = np.searchsorted(raw_durs, elapsed_minutes + horizon_minutes, side="right")
            n_ret = ret_idx - surv_idx
            hazard = (n_ret + 0.15) / (n_surv + 1.0)
            return round(float(np.clip(hazard, 0.005, 0.995)), 4)

        # 正規分布フォールバック
        mu = dist.mean_minutes
        sigma = max(3.0, dist.std_minutes)
        cdf_t = float(stats.norm.cdf(elapsed_minutes, loc=mu, scale=sigma))
        cdf_next = float(stats.norm.cdf(elapsed_minutes + horizon_minutes, loc=mu, scale=sigma))
        survival_t = max(1e-4, 1.0 - cdf_t)

        cond_hazard = (cdf_next - cdf_t) / survival_t
        cond_hazard = max(0.0, min(0.995, cond_hazard))
        return round(float(max(0.005, min(0.995, cond_hazard))), 4)

    def compute_coarse_poi_return_probability(
        self,
        poi: POIClusterSpec,
        elapsed_minutes: float,
        horizon_minutes: float,
    ) -> float:
        """
        従来型の単純統計タイマーベースライン（POIの平均滞在時間のみを参照し、
        ドライバ個性・曜日・時間帯・同乗者文脈を持たない粗い予測確率）
        """
        mu = poi.base_mean_minutes
        sigma = max(5.0, poi.base_std_minutes * 1.35)
        cdf_t = float(stats.norm.cdf(elapsed_minutes, loc=mu, scale=sigma))
        cdf_next = float(stats.norm.cdf(elapsed_minutes + horizon_minutes, loc=mu, scale=sigma))
        survival_t = max(1e-4, 1.0 - cdf_t)
        cond_hazard = (cdf_next - cdf_t) / survival_t
        return round(float(max(0.005, min(0.995, cond_hazard))), 4)

    def update_driver_preference(
        self,
        driver_id: str,
        new_theta: Optional[float] = None,
        new_alpha_range: Optional[float] = None,
        new_beta_comfort: Optional[float] = None,
        allow_enroute_charging: Optional[bool] = None,
    ) -> DriverPreferenceProfile:
        """
        Phase 4 (帰宅後UI) でドライバが再調整した ROC感度閾値 θ や嗜好バランスをプロファイルに反映する
        """
        prof = self.driver_profiles[driver_id]
        if new_theta is not None:
            prof.roc_threshold_theta = round(max(0.05, min(0.95, float(new_theta))), 3)
        if new_alpha_range is not None:
            prof.home_charge_priority_alpha = round(max(0.0, min(1.0, float(new_alpha_range))), 2)
        if new_beta_comfort is not None:
            prof.comfort_priority_beta = round(max(0.0, min(1.0, float(new_beta_comfort))), 2)
        if allow_enroute_charging is not None:
            prof.allow_enroute_charging = bool(allow_enroute_charging)
        return prof

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.history_records])
