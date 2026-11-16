"""
PSMP 4-Phase Full Simulation & EdgeJev vs LightGBM ROC Sensitivity Evaluation Runner (`run_evaluation.py`).
Executes:
  - Phase 1: Historical parking data generation & stay distribution learning across 3 driver personas
  - Phase 2: IG-OFF initial power management & EdgeJev (`choice`, `score`) planning
  - Phase 3: Parked monitoring comparing EdgeJev (`noul`) vs LightGBM (`LGBMClassifier`) wake-up prediction
  - Phase 4: ROC curve / AUC / Brier evaluation, threshold θ sweep, and visual chart generation
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psmp.config import DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator, ROCEvaluationReport


def plot_roc_and_tradeoffs(report: ROCEvaluationReport, out_dir: Path) -> Path:
    """
    EdgeJev vs LightGBM vs Statistical Baseline の ROC感度特性曲線 (Sensitivity vs FPR)、
    閾値 θ スイープ特性、および パレートフロント比較図を生成する
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), dpi=140)
    plt.subplots_adjust(wspace=0.28)

    # --- Panel 1: ROC Curve (EdgeJev vs LightGBM vs Stat Baseline) ---
    ax1 = axes[0]
    ax1.plot(
        report.fpr_curve_jev,
        report.tpr_curve_jev,
        color="#1f77b4",
        lw=2.8,
        label=f"EdgeJev System-1 AI (AUC={report.roc_auc_jev:.4f}, Brier={report.brier_score_jev:.4f})",
    )
    ax1.plot(
        report.fpr_curve_lgbm,
        report.tpr_curve_lgbm,
        color="#2ca02c",
        lw=2.5,
        linestyle="-.",
        label=f"LightGBM GBDT (AUC={report.roc_auc_lgbm:.4f}, Brier={report.brier_score_lgbm:.4f})",
    )
    ax1.plot(
        report.fpr_curve_stat,
        report.tpr_curve_stat,
        color="#ff7f0e",
        lw=1.8,
        linestyle="--",
        label=f"Coarse Stat Timer (AUC={report.roc_auc_stat_baseline:.4f})",
    )
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle=":")

    # 代表動作点 θ=0.25, 0.55 の比較プロット
    for op in report.operating_points:
        if op.theta in (0.25, 0.55):
            ax1.scatter([op.fpr_false_alarm], [op.tpr_sensitivity], color="#1f77b4", s=65, zorder=5)
            ax1.annotate(
                f"Jev θ={op.theta:.2f}",
                (op.fpr_false_alarm, op.tpr_sensitivity),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=8,
                weight="bold",
                color="#1f77b4",
            )
    for op in report.operating_points_lgbm:
        if op.theta in (0.25, 0.55):
            ax1.scatter([op.fpr_false_alarm], [op.tpr_sensitivity], color="#2ca02c", marker="s", s=65, zorder=5)
            ax1.annotate(
                f"LGBM θ={op.theta:.2f}",
                (op.fpr_false_alarm, op.tpr_sensitivity),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                weight="bold",
                color="#2ca02c",
            )

    ax1.set_title("1. ROC Curve: EdgeJev vs LightGBM vs Baseline", fontsize=11.5, weight="bold")
    ax1.set_xlabel("False Positive Rate (FPR: Premature Wakeup Rate)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (TPR: Sensitivity / Ready on Return)", fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.04)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.0)

    # --- Panel 2: Threshold θ Sweep (EdgeJev vs LightGBM Comfort & Energy Saving) ---
    ax2 = axes[1]
    thetas = [op.theta for op in report.operating_points]
    ax2.plot(
        thetas,
        [op.avg_comfort_score for op in report.operating_points],
        "o-",
        color="#1f77b4",
        lw=2.3,
        label="EdgeJev: Ready & Comfort Score",
    )
    ax2.plot(
        thetas,
        [op.avg_comfort_score for op in report.operating_points_lgbm],
        "s--",
        color="#2ca02c",
        lw=2.0,
        label="LightGBM: Ready & Comfort Score",
    )
    ax2.plot(
        thetas,
        [op.energy_saving_vs_always_on_pct for op in report.operating_points],
        "^-",
        color="#9467bd",
        lw=2.2,
        label="EdgeJev: Power Saving vs Always-On (%)",
    )
    ax2.plot(
        thetas,
        [op.energy_saving_vs_always_on_pct for op in report.operating_points_lgbm],
        "v--",
        color="#d62728",
        lw=1.9,
        label="LightGBM: Power Saving vs Always-On (%)",
    )

    ax2.set_title("2. Threshold θ Tuning: EdgeJev vs LightGBM", fontsize=11.5, weight="bold")
    ax2.set_xlabel("Wake-up Trigger Threshold θ (0.10 = High Sensitivity -> 0.90 = Strict Eco)", fontsize=9.5)
    ax2.set_ylabel("Score / Power Saving (%)", fontsize=10)
    ax2.set_ylim(20, 102)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8.0)

    # --- Panel 3: Pareto Front (Energy Wh vs Overall Ready Comfort Score) ---
    ax3 = axes[2]
    ax3.plot(
        [op.avg_energy_wh for op in report.operating_points],
        [op.avg_comfort_score for op in report.operating_points],
        "o-",
        color="#1f77b4",
        lw=2.5,
        label="EdgeJev Pareto Front (Sweep θ)",
    )
    ax3.plot(
        [op.avg_energy_wh for op in report.operating_points_lgbm],
        [op.avg_comfort_score for op in report.operating_points_lgbm],
        "s-.",
        color="#2ca02c",
        lw=2.2,
        label="LightGBM Pareto Front (Sweep θ)",
    )

    comp_df = report.strategy_comparison_df
    for _, row in comp_df.iterrows():
        strat = row["Strategy"]
        if "Stat" in strat:
            ax3.scatter([row["Avg_Energy_Wh"]], [row["Avg_Overall_Ready_Score"]], color="#ff7f0e", marker="D", s=85, label="Statistical Timer Baseline")
        elif "Always-On" in strat:
            ax3.scatter([row["Avg_Energy_Wh"]], [row["Avg_Overall_Ready_Score"]], color="#d62728", marker="X", s=95, label="Always-On (No Saving)")
        elif "No Pre-conditioning" in strat:
            ax3.scatter([row["Avg_Energy_Wh"]], [row["Avg_Overall_Ready_Score"]], color="#7f7f7f", marker="v", s=95, label="No Pre-con (Deep Sleep)")

    ax3.set_title("3. Pareto Front: Energy (Wh) vs Ready Comfort", fontsize=11.5, weight="bold")
    ax3.set_xlabel("Average Parking Energy Consumption (Wh)", fontsize=10)
    ax3.set_ylabel("Overall Departure Ready & Comfort Score (0-100)", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", fontsize=8.0)

    out_path = out_dir / "psmp_roc_and_pareto_evaluation.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_sample_session_timeline(sim_res_jev, sim_res_lgbm, out_dir: Path) -> Path:
    """
    同一駐車セッションにおける EdgeJev vs LightGBM の復帰確率波形 P(wake)、
    車内温度推移、消費電力 W を並べて比較可視化する
    """
    logs_jev = sim_res_jev.timeline_logs
    logs_lgbm = sim_res_lgbm.timeline_logs
    ts = [l.elapsed_minutes for l in logs_jev]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.8), sharex=True, dpi=140)

    # Top: EdgeJev vs LightGBM Wake-up Probability
    ax0 = axes[0]
    ax0.plot(ts, [l.jev_wake_hvac_prob for l in logs_jev], "o-", color="#1f77b4", lw=2.4, markersize=4, label="EdgeJev Noul P(wake_hvac)")
    ax0.plot(ts, [l.lgbm_wake_hvac_prob for l in logs_jev], "s-.", color="#2ca02c", lw=2.2, markersize=4, label="LightGBM Classifier P(wake_hvac)")
    ax0.plot(ts, [l.stat_hazard_prob for l in logs_jev], "--", color="#ff7f0e", lw=1.5, label="Coarse Stat Baseline")
    ax0.plot(ts, [l.effective_threshold_theta for l in logs_jev], ":", color="#d62728", lw=2.0, label="Effective ROC Threshold θ")
    ax0.axvline(sim_res_jev.expected_mean_minutes, color="green", linestyle="-.", alpha=0.6, label=f"Expected Return ({sim_res_jev.expected_mean_minutes:.1f} min)")
    ax0.axvline(sim_res_jev.actual_duration_minutes, color="black", linestyle="-", lw=2.0, label=f"Actual IG-ON Return ({sim_res_jev.actual_duration_minutes:.1f} min)")
    ax0.set_ylabel("Wake Probability", fontsize=10)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title(
        f"EdgeJev vs LightGBM Dynamic Timeline [{sim_res_jev.session_id} | {sim_res_jev.driver_id} | Anomaly: {sim_res_jev.anomaly_type}]",
        fontsize=11.5,
        weight="bold",
    )
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.0, ncol=3)

    # Middle: Cabin Temperature Comparison (EdgeJev vs LightGBM)
    ax1 = axes[1]
    ax1.plot(ts, [l.cabin_temp_c for l in logs_jev], "o-", color="#1f77b4", lw=2.3, markersize=4, label=f"EdgeJev Cabin Temp (End: {sim_res_jev.departure_cabin_temp_c:.1f}°C)")
    ax1.plot(ts, [l.cabin_temp_c for l in logs_lgbm], "s-.", color="#2ca02c", lw=2.1, markersize=4, label=f"LightGBM Cabin Temp (End: {sim_res_lgbm.departure_cabin_temp_c:.1f}°C)")
    ax1.axhline(sim_res_jev.target_cabin_temp_c, color="#d62728", linestyle="--", lw=1.7, label=f"Target Comfort Temp ({sim_res_jev.target_cabin_temp_c:.1f}°C)")
    ax1.set_ylabel("Temperature (°C)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Bottom: Power Draw Comparison (EdgeJev vs LightGBM)
    ax2 = axes[2]
    p_jev = [l.step_power_w for l in logs_jev]
    p_lgbm = [l.step_power_w for l in logs_lgbm]
    ax2.step(ts, p_jev, where="post", color="#1f77b4", lw=2.3, label=f"EdgeJev Power (Total: {sim_res_jev.parking_energy_consumed_wh:.0f} Wh)")
    ax2.step(ts, p_lgbm, where="post", color="#2ca02c", lw=2.0, linestyle="-.", label=f"LightGBM Power (Total: {sim_res_lgbm.parking_energy_consumed_wh:.0f} Wh)")
    ax2.fill_between(ts, p_jev, step="post", color="#1f77b4", alpha=0.15)
    for l in logs_jev:
        if l.re_sleep_triggered:
            ax2.axvline(l.elapsed_minutes, color="#d62728", linestyle="--", lw=1.8)
            ax2.annotate("Re-sleep Guard!", (l.elapsed_minutes, max(p_jev + [500]) * 0.7), color="#d62728", weight="bold", fontsize=8.5)
    ax2.set_xlabel("Elapsed Parking Time from IG-OFF (minutes)", fontsize=10)
    ax2.set_ylabel("Power Draw (W)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / f"psmp_timeline_{sim_res_jev.session_id}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    out_dir = Path("/home/tomokatsu098/develop/PSMP/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [Phase 1] Generating & Learning Driver Parking History ===")
    phase1 = Phase1CollectorLearner()
    all_records = phase1.generate_synthetic_history(num_sessions_per_driver=100, seed=42, include_edge_cases=True)

    train_records = [r for idx, r in enumerate(all_records) if (idx % 100) < 75]
    test_records = [r for idx, r in enumerate(all_records) if (idx % 100) >= 75]
    phase1.fit(train_records)

    print("\n=== [Phase 1-B] Training LightGBM Regressor & Wake-up Classifier ===")
    lgbm_engine = LightGBMDecisionEngine(random_state=42)
    lgbm_summary = lgbm_engine.fit(train_records, phase1)
    print(
        f"LightGBM Trained on {lgbm_summary.num_train_sessions} sessions ({lgbm_summary.num_train_steps} steps) "
        f"in {lgbm_summary.train_time_ms:.1f} ms | Stay Duration RMSE: {lgbm_summary.duration_rmse_min:.2f} min"
    )
    print("Top LightGBM Feature Importances (%):", dict(list(lgbm_summary.feature_importances.items())[:6]))

    print("\n=== [Phase 2 & 3] Initializing EdgeJev & LightGBM Engines + Vehicle Simulator ===")
    jev_engine = JevDecisionEngine(backend_mode="calibrated_local")
    vehicle_sim = VehicleSimulator()
    phase2 = Phase2IGOffPlanner(jev_engine, vehicle_sim)
    phase3 = Phase3ParkedMonitor(phase1, jev_engine, vehicle_sim, lgbm_engine=lgbm_engine)
    phase4 = Phase4PostTripEvaluator(phase1, phase2, phase3)

    print("\n=== [Phase 4] Running EdgeJev vs LightGBM ROC Sensitivity Evaluation ===")
    theta_grid = [round(x, 2) for x in np.linspace(0.10, 0.90, 17)]
    report = phase4.compute_roc_and_threshold_sweep(test_records, theta_grid=theta_grid)

    print("\n[ROC AUC & Calibration Comparison: EdgeJev vs LightGBM]")
    print(f"  - PSMP (EdgeJev Local Decision AI) : ROC AUC = {report.roc_auc_jev:.4f} | Brier Loss = {report.brier_score_jev:.4f}")
    print(f"  - PSMP (LightGBM GBDT Classifier)  : ROC AUC = {report.roc_auc_lgbm:.4f} | Brier Loss = {report.brier_score_lgbm:.4f}")
    print(f"  - Statistical Timer Baseline       : ROC AUC = {report.roc_auc_stat_baseline:.4f}")

    print("\n[5-Strategy Comparison Summary Table]")
    print(report.strategy_comparison_df.to_string(index=False))

    print("\n[Recommended Optimal ROC Threshold θ* by Driver Preference (EdgeJev vs LightGBM)]")
    for drv_id in phase1.driver_profiles:
        prof = phase1.driver_profiles[drv_id]
        th_jev = report.recommended_theta_by_driver[drv_id]
        th_lgbm = report.recommended_theta_lgbm_by_driver[drv_id]
        print(f"  - {drv_id} ({prof.name_ja}): EdgeJev θ*={th_jev:.2f} | LightGBM θ*={th_lgbm:.2f}")

    # Save CSVs & Plots
    op_all_df = pd.DataFrame([vars(op) for op in (report.operating_points + report.operating_points_lgbm)])
    op_all_df.to_csv(out_dir / "roc_operating_points.csv", index=False)
    report.strategy_comparison_df.to_csv(out_dir / "strategy_comparison.csv", index=False)

    roc_chart_path = plot_roc_and_tradeoffs(report, out_dir)
    print(f"\nSaved EdgeJev vs LightGBM ROC & Pareto chart to: {roc_chart_path}")

    summer_events = [e for e in test_records if e.season == "summer" and e.duration_minutes >= 35.0]
    sample_ev = summer_events[0] if summer_events else test_records[0]
    comp_single = phase4.evaluate_single_trip_comparison(sample_ev)
    timeline_chart_path = plot_sample_session_timeline(comp_single["psmp_jev"], comp_single["psmp_lightgbm"], out_dir)
    print(f"Saved Sample Session Timeline chart to: {timeline_chart_path}")

    delay_events = [e for e in test_records if e.anomaly_type == "unexpected_delay" and e.season in ("summer", "winter")]
    if delay_events:
        comp_delay = phase4.evaluate_single_trip_comparison(delay_events[0])
        delay_chart_path = plot_sample_session_timeline(comp_delay["psmp_jev"], comp_delay["psmp_lightgbm"], out_dir)
        print(f"Saved Delayed Session Timeline chart to: {delay_chart_path}")


if __name__ == "__main__":
    main()
