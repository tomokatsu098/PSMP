"""
PSMP Scheme 3: Learning Step & Horizon Evaluation (`evaluate_learning_steps.py`).
Evaluates the learning curve: How prediction accuracy (ROC-AUC, Brier score),
cabin comfort (ready rate), and power savings (Wh) improve as the training horizon
is expanded from 7 days, 14 days, 30 days, 60 days, 90 days, 180 days to 365 days.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psmp.config import DriverPreferenceProfile
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator
from psmp.real_data_loader import RealTripDataLoader


def plot_learning_curves(df_results: pd.DataFrame, out_path: Path) -> None:
    """
    学習日数 (7日〜365日) に対するROC-AUC、Brier Score、快適性・省電力削減率の推移をプロット
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    plt.subplots_adjust(wspace=0.28)

    days = df_results["train_days"].tolist()

    # --- Panel 1: ROC-AUC vs Training Days ---
    ax1 = axes[0]
    ax1.plot(days, df_results["roc_auc_jev"], "o-", color="#1f77b4", lw=2.4, label="EdgeJev Local AI")
    ax1.plot(days, df_results["roc_auc_lgbm"], "s--", color="#2ca02c", lw=2.2, label="LightGBM GBDT")
    ax1.plot(days, df_results["roc_auc_stat"], "^:", color="#ff7f0e", lw=1.8, label="Stat Timer Baseline")
    ax1.set_title("1. ROC-AUC vs Training Days", fontsize=11.5, weight="bold")
    ax1.set_xlabel("Training Duration (Days)", fontsize=10)
    ax1.set_ylabel("ROC AUC", fontsize=10)
    ax1.set_xscale("log")
    ax1.set_xticks(days)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    # --- Panel 2: Brier Calibration Score vs Training Days ---
    ax2 = axes[1]
    ax2.plot(days, df_results["brier_score_jev"], "o-", color="#1f77b4", lw=2.4, label="EdgeJev Brier Loss")
    ax2.plot(days, df_results["brier_score_lgbm"], "s--", color="#2ca02c", lw=2.2, label="LightGBM Brier Loss")
    ax2.set_title("2. Brier Score (Calibration Loss: Lower is Better)", fontsize=11.5, weight="bold")
    ax2.set_xlabel("Training Duration (Days)", fontsize=10)
    ax2.set_ylabel("Brier Score", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_xticks(days)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    # --- Panel 3: Cabin Comfort & Energy Savings vs Training Days ---
    ax3 = axes[1] = axes[2]
    ax3.plot(days, df_results["avg_comfort_score_jev"], "o-", color="#1f77b4", lw=2.2, label="EdgeJev Comfort / Ready Rate (%)")
    ax3.plot(days, df_results["power_saving_pct_jev"], "^-", color="#9467bd", lw=2.2, label="EdgeJev Power Saving vs Always-On (%)")
    ax3.set_title("3. Ready Rate & Power Savings (θ=0.50)", fontsize=11.5, weight="bold")
    ax3.set_xlabel("Training Duration (Days)", fontsize=10)
    ax3.set_ylabel("Score / Saving Ratio (%)", fontsize=10)
    ax3.set_xscale("log")
    ax3.set_xticks(days)
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def get_data_path(base_name: str) -> str:
    base_dir = Path(__file__).resolve().parent
    local_gz = base_dir / "data" / f"{base_name}.csv.gz"
    if local_gz.exists():
        return str(local_gz)
    local_csv = base_dir / "data" / f"{base_name}.csv"
    if local_csv.exists():
        return str(local_csv)
    net_path = Path(f"/usr/local/NetworkShare/PSMP_workspace/{base_name}.csv")
    if net_path.exists():
        return str(net_path)
    return str(local_gz)


def run_learning_step_evaluation():
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = get_data_path("syn548_profile_v4_train")
    eval_csv = get_data_path("syn548_profile_v4_eval")

    # 代表車両（Group Aから2台、Group Bから2台）
    target_vehicles = ["A00001", "A00002", "B00001", "B00002"]
    horizons = [7, 14, 30, 60, 90, 180, 365]

    print("=====================================================================")
    print("PSMP Scheme 3: Training Step Horizon Evaluation (Learning Curve)")
    print(f"Target Vehicles: {target_vehicles}")
    print(f"Step Horizons  : {horizons} days")
    print("=====================================================================\n")

    loader = RealTripDataLoader(train_csv, eval_csv)

    # 評価用データセットの抽出（固定テストセット：出先駐車セッション）
    print("Loading Common Evaluation Set (2026-04-01 ~ 2026-10-01)...")
    raw_eval_records, eval_pois = loader.get_eval_records(vehicle_ids=target_vehicles, out_of_home_only=True)
    
    # 計算時間と精度のバランスのため、各車両15セッション（計60セッション）にサンプリング
    eval_records_sample = []
    for vid in target_vehicles:
        v_recs = [r for r in raw_eval_records if r.driver_id == vid]
        eval_records_sample.extend(v_recs[:15])
    print(f"Loaded {len(eval_records_sample)} evaluation sessions across {len(target_vehicles)} vehicles.\n")

    results = []

    for h in horizons:
        print(f"--- Running Evaluation for Horizon = {h:3d} Days ---")
        train_records, train_pois = loader.get_train_records(
            vehicle_ids=target_vehicles,
            max_train_days=h,
            out_of_home_only=True,
        )

        phase1 = Phase1CollectorLearner()
        phase1.poi_clusters.update(eval_pois)
        phase1.poi_clusters.update(train_pois)

        for vid in target_vehicles:
            phase1.driver_profiles[vid] = DriverPreferenceProfile(
                driver_id=vid,
                name_ja=f"車両_{vid}",
                description_ja=f"実データ {vid}",
                home_charge_priority_alpha=0.60,
                comfort_priority_beta=0.60,
                allow_enroute_charging=False,
                roc_threshold_theta=0.50,
                min_home_arrival_soc_pct=15.0,
                battery_capacity_kwh=60.0,
                efficiency_km_per_kwh=6.5,
            )

        phase1.fit(train_records)

        # LightGBM 学習
        lgbm_engine = LightGBMDecisionEngine(random_state=42)
        try:
            lgbm_summary = lgbm_engine.fit(train_records, phase1)
            duration_rmse = lgbm_summary.duration_rmse_min
        except Exception as e:
            print(f"  [Warning] LightGBM training failed for horizon {h}: {e}")
            duration_rmse = 999.0

        jev_engine = JevDecisionEngine(backend_mode="calibrated_local")
        vehicle_sim = VehicleSimulator()
        phase2 = Phase2IGOffPlanner(jev_engine, vehicle_sim)
        phase3 = Phase3ParkedMonitor(phase1, jev_engine, vehicle_sim, lgbm_engine=lgbm_engine)
        phase4 = Phase4PostTripEvaluator(phase1, phase2, phase3)

        theta_grid = [0.30, 0.50, 0.70]
        report = phase4.compute_roc_and_threshold_sweep(eval_records_sample, theta_grid=theta_grid)

        # 代表動作点 θ=0.50 の性能値を取得
        op_theta_05 = [op for op in report.operating_points if abs(op.theta - 0.50) < 1e-4]
        if op_theta_05:
            comfort_val = op_theta_05[0].avg_comfort_score
            saving_pct = op_theta_05[0].energy_saving_vs_always_on_pct
            energy_wh = op_theta_05[0].avg_energy_wh
        else:
            comfort_val = 0.0
            saving_pct = 0.0
            energy_wh = 0.0

        res_row = {
            "train_days": h,
            "train_sessions_count": len(train_records),
            "roc_auc_jev": round(report.roc_auc_jev, 4),
            "brier_score_jev": round(report.brier_score_jev, 4),
            "roc_auc_lgbm": round(report.roc_auc_lgbm, 4),
            "brier_score_lgbm": round(report.brier_score_lgbm, 4),
            "roc_auc_stat": round(report.roc_auc_stat_baseline, 4),
            "avg_comfort_score_jev": round(comfort_val, 2),
            "power_saving_pct_jev": round(saving_pct, 2),
            "avg_energy_consumed_wh_jev": round(energy_wh, 1),
            "lgbm_duration_rmse_min": round(duration_rmse, 2),
        }
        results.append(res_row)
        print(
            f"  [Result {h:3d}d] Train Sessions: {len(train_records):4d} | "
            f"EdgeJev AUC: {report.roc_auc_jev:.4f} (Brier: {report.brier_score_jev:.4f}) | "
            f"LGBM AUC: {report.roc_auc_lgbm:.4f} | Comfort: {comfort_val:.1f}% | Saving: {saving_pct:.1f}%\n"
        )

    df_results = pd.DataFrame(results)
    csv_out = out_dir / "learning_step_evaluation_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"Saved evaluation metrics table to: {csv_out}")

    chart_out = out_dir / "learning_curve_steps.png"
    plot_learning_curves(df_results, chart_out)
    print(f"Saved Learning Curve charts to: {chart_out}")

    print("\n=== Final Summary Table ===")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    run_learning_step_evaluation()
