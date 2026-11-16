"""
Phase 4: Post-Trip Evaluator & ROC Sensitivity Tuning Engine (`phase4_post_trip_evaluator.py`)
４．帰宅後、効果を可視化し、かつ、次回に向けて再調整できるUI部位（評価・最適化バックエンド）
  - 帰宅後の省電力効果（常時稼働比の削減電力量Wh）、入車時快適度・レディ達成率、無充電帰宅達成率を算出
  - EdgeJev (`edgejev` ローカル判断特化型AI) と LightGBM (`lightgbm` GBDT) および統計ベースラインの
    ROC感度特性曲線 (TPR vs FPR)・AUC・閾値 θ スイープ特性・特徴量重要度を比較評価
  - ドライバごとの個人嗜好（電費優先度 α vs 快適性優先度 β）に応じた最適動作点 θ* を算出し、
    Phase 1 のプロファイルへフィードバックする
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, roc_curve

from psmp.config import DriverPreferenceProfile
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import ParkingEventRecord, Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import ParkingSessionSimulationResult, Phase3ParkedMonitor


@dataclass
class ROCOperatingPointMetrics:
    """
    特定の判定閾値 θ における ROC 指標（感度・偽陽性率）およびシステム KPI
    """
    model_name: str                       # "EdgeJev" | "LightGBM"
    theta: float
    tpr_sensitivity: float                # 真陽性率 (帰還直前ウィンドウで正しくレディ起動できた率)
    fpr_false_alarm: float                # 偽陽性率 (まだ戻らない時間帯にフライング稼働した率)
    specificity: float                    # 特異度 (1 - FPR)
    youden_j: float                       # Youden Index (TPR - FPR)
    session_ready_success_rate: float     # 入車時に快適＆レディ状態であったセッション割合 (%)
    avg_comfort_score: float              # 入車時の平均快適度スコア (0~100)
    avg_energy_wh: float                  # 1セッションあたり平均消費電力量 (Wh)
    energy_saving_vs_always_on_pct: float # 常時稼働比の省電力削減率 (%)
    home_without_charge_rate: float       # 自宅まで無充電で帰還できた割合 (%)
    utility_score: float                  # ドライバ嗜好を加味した総合効用スコア


@dataclass
class ROCEvaluationReport:
    """
    EdgeJev vs LightGBM vs Statistical Baseline の ROC曲線・AUC・手法間比較レポート
    """
    roc_auc_jev: float
    roc_auc_lgbm: float
    roc_auc_stat_baseline: float
    brier_score_jev: float
    brier_score_lgbm: float
    fpr_curve_jev: List[float]
    tpr_curve_jev: List[float]
    thresholds_jev: List[float]
    fpr_curve_lgbm: List[float]
    tpr_curve_lgbm: List[float]
    thresholds_lgbm: List[float]
    fpr_curve_stat: List[float]
    tpr_curve_stat: List[float]
    operating_points: List[ROCOperatingPointMetrics]       # EdgeJev 閾値スイープ
    operating_points_lgbm: List[ROCOperatingPointMetrics]  # LightGBM 閾値スイープ
    recommended_theta_by_driver: Dict[str, float]
    recommended_theta_lgbm_by_driver: Dict[str, float]
    strategy_comparison_df: pd.DataFrame
    lgbm_feature_importances: Dict[str, float] = field(default_factory=dict)


class Phase4PostTripEvaluator:
    """
    【Phase 4】帰宅後の効果評価・EdgeJev vs LightGBM ROC感度特性解析・次回向け閾値 θ 再調整エンジン
    """

    def __init__(
        self,
        phase1: Phase1CollectorLearner,
        phase2: Phase2IGOffPlanner,
        phase3: Phase3ParkedMonitor,
    ):
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3

    def evaluate_single_trip_comparison(
        self,
        event: ParkingEventRecord,
        custom_theta: Optional[float] = None,
    ) -> Dict[str, ParkingSessionSimulationResult]:
        """
        1回の駐車〜帰宅トリップについて、以下の5方式を同一条件で実行して帰宅後比較サマリーを生成する：
          1. `psmp_jev`: PSMP (EdgeJev ローカル判断特化型AI + ROC閾値制御)
          2. `psmp_lightgbm`: PSMP (LightGBM GBDT + ROC閾値制御)
          3. `stat_baseline`: 統計タイマーベースライン
          4. `always_on`: 常時稼働（省電力なし・快適性上限）
          5. `no_precon`: 完全スリープ放置（事前復帰なし・省電力上限）
        """
        profile = self.phase1.driver_profiles[event.driver_id]
        poi, stay_dist = self.phase1.predict_stay_distribution(
            driver_id=event.driver_id,
            lat=event.lat,
            lon=event.lon,
            park_start_dt=event.park_start_dt,
            is_holiday=event.is_holiday,
            has_passengers=event.has_passengers,
        )
        init_plan, init_state = self.phase2.create_initial_plan(event, profile, poi, stay_dist)

        results: Dict[str, ParkingSessionSimulationResult] = {}
        for strat in ["psmp_jev", "psmp_lightgbm", "stat_baseline", "always_on", "no_precon"]:
            results[strat] = self.phase3.run_parked_session(
                event=event,
                profile=profile,
                initial_plan=init_plan,
                initial_env_state=init_state,
                override_theta=custom_theta,
                control_strategy=strat,
            )
        return results

    def _sweep_thresholds_for_strategy(
        self,
        cached_plans: List[Tuple[Any, ...]],
        theta_grid: List[float],
        control_strategy: str,
        model_label: str,
        avg_always_on_wh: float,
    ) -> List[ROCOperatingPointMetrics]:
        op_metrics_list: List[ROCOperatingPointMetrics] = []
        for theta in theta_grid:
            tp, fp, tn, fn = 0, 0, 0, 0
            ready_hits = 0
            comforts = []
            energies = []
            home_ok_hits = 0

            for ev, prof, plan, init_state in cached_plans:
                sim_res = self.phase3.run_parked_session(
                    ev, prof, plan, init_state, override_theta=theta, control_strategy=control_strategy
                )
                ready_hits += 1 if sim_res.is_ready_success else 0
                comforts.append(sim_res.overall_ready_score)
                energies.append(sim_res.parking_energy_consumed_wh)
                home_ok_hits += 1 if sim_res.can_return_home_without_charging else 0

                for sl in sim_res.timeline_logs:
                    gt = sl.ground_truth_should_wake_hvac
                    pred = sl.predicted_wake_hvac
                    if gt and pred:
                        tp += 1
                    elif (not gt) and pred:
                        fp += 1
                    elif (not gt) and (not pred):
                        tn += 1
                    else:
                        fn += 1

            tpr_val = tp / max(1, tp + fn)
            fpr_val = fp / max(1, fp + tn)
            spec_val = 1.0 - fpr_val
            youden = tpr_val - fpr_val

            avg_e = float(np.mean(energies))
            saving_pct = (1.0 - avg_e / max(1.0, avg_always_on_wh)) * 100.0
            ready_rate = (ready_hits / len(cached_plans)) * 100.0
            avg_comf = float(np.mean(comforts))
            home_rate = (home_ok_hits / len(cached_plans)) * 100.0

            util = 0.50 * (avg_comf / 100.0) + 0.30 * (saving_pct / 100.0) + 0.20 * youden

            op_metrics_list.append(
                ROCOperatingPointMetrics(
                    model_name=model_label,
                    theta=round(float(theta), 2),
                    tpr_sensitivity=round(tpr_val, 4),
                    fpr_false_alarm=round(fpr_val, 4),
                    specificity=round(spec_val, 4),
                    youden_j=round(youden, 4),
                    session_ready_success_rate=round(ready_rate, 1),
                    avg_comfort_score=round(avg_comf, 1),
                    avg_energy_wh=round(avg_e, 1),
                    energy_saving_vs_always_on_pct=round(saving_pct, 1),
                    home_without_charge_rate=round(home_rate, 1),
                    utility_score=round(util, 4),
                )
            )
        return op_metrics_list

    def _compute_recommended_thetas(
        self, op_metrics_list: List[ROCOperatingPointMetrics]
    ) -> Dict[str, float]:
        recommended_thetas: Dict[str, float] = {}
        for drv_id, prof in self.phase1.driver_profiles.items():
            best_th = prof.roc_threshold_theta
            best_u = -1e9
            target_center_theta = 0.45 + 0.35 * (prof.home_charge_priority_alpha - prof.comfort_priority_beta)
            for op in op_metrics_list:
                youden_term = op.tpr_sensitivity - op.fpr_false_alarm
                pref_alignment = -0.95 * ((op.theta - target_center_theta) ** 2)
                drv_u = 0.55 * youden_term + pref_alignment
                if drv_u > best_u:
                    best_u = drv_u
                    best_th = op.theta
            recommended_thetas[drv_id] = best_th
        return recommended_thetas

    def compute_roc_and_threshold_sweep(
        self,
        test_events: List[ParkingEventRecord],
        theta_grid: Optional[List[float]] = None,
        driver_filter: Optional[str] = None,
    ) -> ROCEvaluationReport:
        """
        評価用駐車セッション群に対して：
          1. EdgeJev vs LightGBM vs 統計ベースライン の連続 ROC曲線・AUC・Brierスコアを計算
          2. 閾値 θ をスイープした際の EdgeJev と LightGBM それぞれのパレートトレードオフ特性を算出
          3. 各ドライバの個人嗜好 (α: 自宅充電こだわり度, β: 快適性優先度) に最適な θ* を導出
        """
        events = [
            e for e in test_events
            if (driver_filter is None or e.driver_id == driver_filter)
        ]
        if not events:
            raise ValueError("No test events matched the filter.")

        if theta_grid is None:
            theta_grid = [round(x, 2) for x in np.linspace(0.10, 0.90, 17)]

        y_true_steps: List[int] = []
        y_score_jev: List[float] = []
        y_score_lgbm: List[float] = []
        y_score_stat: List[float] = []

        always_on_results: List[ParkingSessionSimulationResult] = []
        no_precon_results: List[ParkingSessionSimulationResult] = []
        default_jev_results: List[ParkingSessionSimulationResult] = []
        default_lgbm_results: List[ParkingSessionSimulationResult] = []
        default_stat_results: List[ParkingSessionSimulationResult] = []

        cached_plans = []
        for ev in events:
            prof = self.phase1.driver_profiles[ev.driver_id]
            poi, stay_dist = self.phase1.predict_stay_distribution(
                ev.driver_id, ev.lat, ev.lon, ev.park_start_dt, ev.is_holiday, ev.has_passengers
            )
            plan, init_state = self.phase2.create_initial_plan(ev, prof, poi, stay_dist)
            cached_plans.append((ev, prof, plan, init_state))

            res_jev = self.phase3.run_parked_session(
                ev, prof, plan, init_state, control_strategy="psmp_jev"
            )
            res_lgbm = self.phase3.run_parked_session(
                ev, prof, plan, init_state, control_strategy="psmp_lightgbm"
            )
            res_stat = self.phase3.run_parked_session(
                ev, prof, plan, init_state, control_strategy="stat_baseline"
            )
            res_ao = self.phase3.run_parked_session(
                ev, prof, plan, init_state, control_strategy="always_on"
            )
            res_np = self.phase3.run_parked_session(
                ev, prof, plan, init_state, control_strategy="no_precon"
            )

            default_jev_results.append(res_jev)
            default_lgbm_results.append(res_lgbm)
            default_stat_results.append(res_stat)
            always_on_results.append(res_ao)
            no_precon_results.append(res_np)

            for step_log in res_jev.timeline_logs:
                y_true_steps.append(1 if step_log.ground_truth_should_wake_hvac else 0)
                y_score_jev.append(step_log.jev_wake_hvac_prob)
                y_score_lgbm.append(step_log.lgbm_wake_hvac_prob)
                y_score_stat.append(step_log.stat_hazard_prob)

        # ROC Curve, AUC, Brier Score (確率較正精度: 低いほど優れる)
        fpr_jev, tpr_jev, th_jev = roc_curve(y_true_steps, y_score_jev)
        roc_auc_jev = float(auc(fpr_jev, tpr_jev))
        brier_jev = float(brier_score_loss(y_true_steps, y_score_jev))

        fpr_lgbm, tpr_lgbm, th_lgbm = roc_curve(y_true_steps, y_score_lgbm)
        roc_auc_lgbm = float(auc(fpr_lgbm, tpr_lgbm))
        brier_lgbm = float(brier_score_loss(y_true_steps, y_score_lgbm))

        fpr_stat, tpr_stat, _ = roc_curve(y_true_steps, y_score_stat)
        roc_auc_stat = float(auc(fpr_stat, tpr_stat))

        avg_always_on_wh = float(np.mean([r.parking_energy_consumed_wh for r in always_on_results]))

        # EdgeJev と LightGBM それぞれの閾値 θ スイープ
        op_jev_list = self._sweep_thresholds_for_strategy(
            cached_plans, theta_grid, "psmp_jev", "EdgeJev", avg_always_on_wh
        )
        op_lgbm_list = self._sweep_thresholds_for_strategy(
            cached_plans, theta_grid, "psmp_lightgbm", "LightGBM", avg_always_on_wh
        )

        rec_thetas_jev = self._compute_recommended_thetas(op_jev_list)
        rec_thetas_lgbm = self._compute_recommended_thetas(op_lgbm_list)

        # 5方式比較サマリーテーブル作成
        comp_rows = []
        for label, res_list, auc_val, brier_val in [
            ("PSMP (EdgeJev Local Decision AI)", default_jev_results, roc_auc_jev, brier_jev),
            ("PSMP (LightGBM GBDT Classifier)", default_lgbm_results, roc_auc_lgbm, brier_lgbm),
            ("Statistical Timer Baseline", default_stat_results, roc_auc_stat, None),
            ("Always-On (No Power Saving)", always_on_results, None, None),
            ("No Pre-conditioning (Deep Sleep Only)", no_precon_results, None, None),
        ]:
            avg_wh = float(np.mean([r.parking_energy_consumed_wh for r in res_list]))
            comp_rows.append(
                {
                    "Strategy": label,
                    "ROC_AUC": round(auc_val, 4) if auc_val is not None else "-",
                    "Brier_Calibration_Loss": round(brier_val, 4) if brier_val is not None else "-",
                    "Avg_Energy_Wh": round(avg_wh, 1),
                    "Energy_Saving_vs_AlwaysOn_Pct": round((1.0 - avg_wh / max(1.0, avg_always_on_wh)) * 100.0, 1),
                    "Avg_Overall_Ready_Score": round(float(np.mean([r.overall_ready_score for r in res_list])), 1),
                    "Avg_Cabin_Comfort_Score": round(float(np.mean([r.cabin_comfort_score for r in res_list])), 1),
                    "Ready_Success_Rate_Pct": round(
                        (sum(1 for r in res_list if r.is_ready_success) / len(res_list)) * 100.0, 1
                    ),
                    "Home_NoCharge_Success_Pct": round(
                        (sum(1 for r in res_list if r.can_return_home_without_charging) / len(res_list)) * 100.0, 1
                    ),
                    "Avg_Temp_Error_C": round(float(np.mean([r.departure_temp_error_c for r in res_list])), 2),
                }
            )

        feat_imp = {}
        if self.phase3.lgbm is not None and self.phase3.lgbm.fit_summary is not None:
            feat_imp = self.phase3.lgbm.fit_summary.feature_importances

        return ROCEvaluationReport(
            roc_auc_jev=round(roc_auc_jev, 4),
            roc_auc_lgbm=round(roc_auc_lgbm, 4),
            roc_auc_stat_baseline=round(roc_auc_stat, 4),
            brier_score_jev=round(brier_jev, 4),
            brier_score_lgbm=round(brier_lgbm, 4),
            fpr_curve_jev=[round(float(x), 4) for x in fpr_jev],
            tpr_curve_jev=[round(float(x), 4) for x in tpr_jev],
            thresholds_jev=[round(float(x), 4) for x in th_jev],
            fpr_curve_lgbm=[round(float(x), 4) for x in fpr_lgbm],
            tpr_curve_lgbm=[round(float(x), 4) for x in tpr_lgbm],
            thresholds_lgbm=[round(float(x), 4) for x in th_lgbm],
            fpr_curve_stat=[round(float(x), 4) for x in fpr_stat],
            tpr_curve_stat=[round(float(x), 4) for x in tpr_stat],
            operating_points=op_jev_list,
            operating_points_lgbm=op_lgbm_list,
            recommended_theta_by_driver=rec_thetas_jev,
            recommended_theta_lgbm_by_driver=rec_thetas_lgbm,
            strategy_comparison_df=pd.DataFrame(comp_rows),
            lgbm_feature_importances=feat_imp,
        )
