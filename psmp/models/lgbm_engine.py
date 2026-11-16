"""
LightGBM Decision Engine (`lgbm_engine.py`).
Trains and serves LightGBM GBDT models for PSMP:
  1. `LGBMRegressor`: Predicts parking stay duration (minutes) from Phase 1 context features
     (driver_id, poi_id, lat, lon, day_of_week, is_holiday, hour_of_day, season, ambient_temp_c, has_passengers, initial_soc_pct)
  2. `LGBMClassifier`: Directly predicts step-level wake-up probability P(wake_hvac) ∈ [0, 1]
     given elapsed time, context features, cabin/ambient temperature, SoC margin, and stay prediction.
Provides a unified inference interface and feature importance analysis for direct comparison with EdgeJev.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from psmp.config import DEVICE_SPECS, DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS
from psmp.phase1_collector_learner import ParkingEventRecord, Phase1CollectorLearner


DRIVER_ID_MAP = {
    "DRV_A_ECO_HOME": 0,
    "DRV_B_COMFORT_FIRST": 1,
    "DRV_C_PHEV_BALANCED": 2,
}

POI_ID_MAP = {
    "POI_CONV_01": 0,
    "POI_SUPER_02": 1,
    "POI_REST_03": 2,
    "POI_OFFICE_04": 3,
    "POI_GYM_05": 4,
}

SEASON_MAP = {
    "summer": 0,
    "winter": 1,
    "spring_autumn": 2,
}


@dataclass
class LGBMFitSummary:
    num_train_sessions: int
    num_train_steps: int
    train_time_ms: float
    duration_rmse_min: float
    feature_importances: Dict[str, float]


class LightGBMDecisionEngine:
    """
    LightGBM (勾配ブースティング決定木) による滞在時間回帰＆事前復帰確率分類エンジン。
    EdgeJev (ローカル判断特化型AI) と同一の入力コンテキスト・タイムステップから
    復帰確率 P(wake) ∈ [0, 1] を推論し、ROC感度特性・省電力・快適性を直接比較可能にする。
    """

    STEP_FEATURE_NAMES = [
        "driver_code",
        "poi_code",
        "is_holiday",
        "hour_of_day",
        "season_code",
        "ambient_temp_c",
        "has_passengers",
        "elapsed_minutes",
        "lgbm_pred_stay_minutes",
        "remaining_est_minutes",
        "z_Remaining_ratio",
        "stat_cond_hazard_prob",
        "soc_margin_pct",
        "alpha_range_priority",
        "beta_comfort_priority",
    ]

    SESSION_FEATURE_NAMES = [
        "driver_code",
        "poi_code",
        "day_of_week",
        "is_holiday",
        "hour_of_day",
        "season_code",
        "ambient_temp_c",
        "has_passengers",
        "distance_to_home_km",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.duration_regressor: Optional[lgb.LGBMRegressor] = None
        self.wakeup_classifier: Optional[lgb.LGBMClassifier] = None
        self.fit_summary: Optional[LGBMFitSummary] = None
        self.is_fitted: bool = False

    def _encode_session_features(self, rec: ParkingEventRecord) -> List[float]:
        return [
            float(DRIVER_ID_MAP.get(rec.driver_id, 0)),
            float(POI_ID_MAP.get(rec.poi_id, 0)),
            float(rec.day_of_week),
            1.0 if rec.is_holiday else 0.0,
            float(rec.hour_of_day),
            float(SEASON_MAP.get(rec.season, 2)),
            float(rec.ambient_temp_c),
            1.0 if rec.has_passengers else 0.0,
            float(rec.distance_to_home_km),
        ]

    def fit(
        self,
        train_records: List[ParkingEventRecord],
        phase1_learner: Phase1CollectorLearner,
        dt_minutes: float = 2.0,
    ) -> LGBMFitSummary:
        """
        Phase 1 の過去駐車履歴 (`train_records`) を用いて：
          1. 滞在時間予測モデル (`LGBMRegressor`) を学習
          2. 各駐車セッションをタイムステップ展開した教師データで復帰分類器 (`LGBMClassifier`) を学習
        """
        t0 = time.perf_counter()

        # 1. Session-level Duration Regressor の学習
        X_sess = np.array([self._encode_session_features(r) for r in train_records], dtype=np.float32)
        y_sess = np.array([r.duration_minutes for r in train_records], dtype=np.float32)

        self.duration_regressor = lgb.LGBMRegressor(
            n_estimators=90,
            max_depth=5,
            num_leaves=15,
            learning_rate=0.07,
            subsample_for_bin=200,
            min_child_samples=5,
            random_state=self.random_state,
            verbose=-1,
        )
        self.duration_regressor.fit(X_sess, y_sess)
        pred_stays = self.duration_regressor.predict(X_sess)
        rmse = float(np.sqrt(np.mean((pred_stays - y_sess) ** 2)))

        # 2. Step-level Wake-up Binary Classifier の学習データ構築
        X_steps: List[List[float]] = []
        y_steps: List[int] = []
        lead_window = DEVICE_SPECS.hvac_lead_time_minutes + 4.0

        for idx, rec in enumerate(train_records):
            prof = phase1_learner.driver_profiles[rec.driver_id]
            _, stay_dist = phase1_learner.predict_stay_distribution(
                rec.driver_id, rec.lat, rec.lon, rec.park_start_dt, rec.is_holiday, rec.has_passengers
            )
            est_stay_lgbm = float(pred_stays[idx])

            req_home_kwh = rec.distance_to_home_km / prof.efficiency_km_per_kwh
            req_home_soc = (req_home_kwh / prof.battery_capacity_kwh) * 100.0 + prof.min_home_arrival_soc_pct
            soc_margin = rec.initial_soc_pct - req_home_soc

            t = 0.0
            while t < rec.duration_minutes:
                rem_actual = rec.duration_minutes - t
                gt_wake = 1 if rem_actual <= lead_window else 0

                cond_prob = phase1_learner.compute_conditional_return_probability(
                    stay_dist, elapsed_minutes=t, horizon_minutes=DEVICE_SPECS.hvac_lead_time_minutes
                )
                rem_est = est_stay_lgbm - t
                z_ratio = (t + DEVICE_SPECS.hvac_lead_time_minutes - est_stay_lgbm) / max(4.0, stay_dist.std_minutes)

                step_feat = [
                    float(DRIVER_ID_MAP.get(rec.driver_id, 0)),
                    float(POI_ID_MAP.get(rec.poi_id, 0)),
                    1.0 if rec.is_holiday else 0.0,
                    float(rec.hour_of_day),
                    float(SEASON_MAP.get(rec.season, 2)),
                    float(rec.ambient_temp_c),
                    1.0 if rec.has_passengers else 0.0,
                    float(t),
                    round(est_stay_lgbm, 2),
                    round(rem_est, 2),
                    round(z_ratio, 3),
                    float(cond_prob),
                    round(soc_margin, 2),
                    float(prof.home_charge_priority_alpha),
                    float(prof.comfort_priority_beta),
                ]
                X_steps.append(step_feat)
                y_steps.append(gt_wake)
                t += dt_minutes

        X_steps_arr = np.array(X_steps, dtype=np.float32)
        y_steps_arr = np.array(y_steps, dtype=np.int32)

        self.wakeup_classifier = lgb.LGBMClassifier(
            n_estimators=110,
            max_depth=5,
            num_leaves=15,
            learning_rate=0.06,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            verbose=-1,
        )
        self.wakeup_classifier.fit(X_steps_arr, y_steps_arr)
        self.is_fitted = True

        train_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        importances = self.wakeup_classifier.feature_importances_
        total_imp = max(1, int(np.sum(importances)))
        imp_dict = {
            name: round(float(val) / total_imp * 100.0, 2)
            for name, val in sorted(
                zip(self.STEP_FEATURE_NAMES, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        self.fit_summary = LGBMFitSummary(
            num_train_sessions=len(train_records),
            num_train_steps=len(X_steps),
            train_time_ms=train_ms,
            duration_rmse_min=round(rmse, 2),
            feature_importances=imp_dict,
        )
        return self.fit_summary

    def predict_stay_duration(self, rec: ParkingEventRecord) -> float:
        if not self.is_fitted or self.duration_regressor is None:
            raise RuntimeError("LightGBMDecisionEngine is not fitted yet.")
        x = np.array([self._encode_session_features(rec)], dtype=np.float32)
        pred = float(self.duration_regressor.predict(x)[0])
        return round(max(5.0, pred), 2)

    def predict_wakeup_probability(
        self,
        rec: ParkingEventRecord,
        elapsed_minutes: float,
        expected_std_minutes: float,
        stat_cond_hazard_prob: float,
        soc_margin_pct: float,
        alpha_range: float,
        beta_comfort: float,
        cached_pred_stay: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        現在のタイムステップにおける LightGBM の復帰予測確率 P(wake_hvac) ∈ [0, 1] と推論レイテンシ(ms)を返す
        """
        if not self.is_fitted or self.wakeup_classifier is None:
            raise RuntimeError("LightGBMDecisionEngine is not fitted yet.")

        t0 = time.perf_counter()
        est_stay = cached_pred_stay if cached_pred_stay is not None else self.predict_stay_duration(rec)
        rem_est = est_stay - elapsed_minutes
        z_ratio = (elapsed_minutes + DEVICE_SPECS.hvac_lead_time_minutes - est_stay) / max(4.0, expected_std_minutes)

        feat = np.array(
            [[
                float(DRIVER_ID_MAP.get(rec.driver_id, 0)),
                float(POI_ID_MAP.get(rec.poi_id, 0)),
                1.0 if rec.is_holiday else 0.0,
                float(rec.hour_of_day),
                float(SEASON_MAP.get(rec.season, 2)),
                float(rec.ambient_temp_c),
                1.0 if rec.has_passengers else 0.0,
                float(elapsed_minutes),
                round(est_stay, 2),
                round(rem_est, 2),
                round(z_ratio, 3),
                float(stat_cond_hazard_prob),
                round(soc_margin_pct, 2),
                float(alpha_range),
                float(beta_comfort),
            ]],
            dtype=np.float32,
        )
        prob = float(self.wakeup_classifier.predict_proba(feat)[0, 1])
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return round(max(0.005, min(0.995, prob)), 4), latency_ms
