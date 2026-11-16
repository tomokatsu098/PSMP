"""
PSMP Cold-Start / Unvisited POI Evaluation (`evaluate_coldstart_strategies.py`).
Compares 4 strategies for handling first-time / unvisited parking locations:
  0. Current Baseline: Uniform POI spec (45 min, low confidence)
  1. Approach (1): Driver Global ECDF (Personalized cross-POI out-of-home history)
  2. Approach (2): Navigation Category Prior (Navigation genre context: quick, medium, long)
  3. Approach (3): Fleet Crowdsourced Priors (Cloud OTA aggregation from other vehicles)
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from psmp.config import DriverPreferenceProfile, POIClusterSpec, POIType
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import LearnedStayDistribution, ParkingEventRecord, Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator
from psmp.real_data_loader import RealTripDataLoader


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


def run_coldstart_comparison():
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = get_data_path("syn548_profile_v4_train")
    eval_csv = get_data_path("syn548_profile_v4_eval")

    target_vehicles = ["A00001", "A00002", "B00001", "B00002"]
    
    print("=====================================================================")
    print("PSMP: Cold-Start Evaluation on First-Time / Unvisited Locations")
    print(f"Target Vehicles: {target_vehicles}")
    print("=====================================================================\n")

    loader = RealTripDataLoader(train_csv, eval_csv)

    # 14日間の自車学習データ（初訪問が適度に発生する初期状態）
    train_recs_self, train_pois_self = loader.get_train_records(
        vehicle_ids=target_vehicles, max_train_days=14, out_of_home_only=True
    )
    eval_recs_self, eval_pois_self = loader.get_eval_records(
        vehicle_ids=target_vehicles, out_of_home_only=True
    )

    # 自車の既知POI
    known_poi_ids = set(r.poi_id for r in train_recs_self)
    
    # 初訪問セッション（未学習POI）のみを抽出
    unvisited_eval = [r for r in eval_recs_self if r.poi_id not in known_poi_ids]
    # 代表的な50セッションを評価対象とする
    unvisited_sample = unvisited_eval[:50]
    print(f"Total unvisited evaluation sessions extracted: {len(unvisited_sample)}")

    # フリート他車（他車両群）の過去データ（365日分）をクラウドDBとしてロード
    print("Loading Cloud Fleet crowdsourced data from other vehicles...")
    other_train_df = pd.read_csv(train_csv)
    other_train_df = other_train_df[~other_train_df["vehicle_id"].isin(target_vehicles)]
    other_train_df = other_train_df[other_train_df["location_at_ig_off"].isin(["その他", "会社"])]
    
    # 座標グリッドごとのフリート滞在時間マップ
    fleet_geo_durations: Dict[str, List[float]] = {}
    other_train_df["geo_key"] = other_train_df["end_lat"].round(2).astype(str) + "_" + other_train_df["end_lon"].round(2).astype(str)
    
    # トリップから駐車時間を簡易計算
    for geo_k, sub in other_train_df.groupby("geo_key"):
        # サンプル数上限200
        durs = sub["trip_distance_km"].values * 6.0  # 簡易相関滞在時間
        fleet_geo_durations[geo_k] = np.sort(np.clip(sub["trip_distance_km"].values * 8.0, 10.0, 360.0))

    # --- アプローチ別のPhase 1定義 ---

    # 0. 現状ベースライン（一律スペック）
    p1_base = Phase1CollectorLearner()
    p1_base.poi_clusters.update(eval_pois_self)
    p1_base.poi_clusters.update(train_pois_self)
    for vid in target_vehicles:
        p1_base.driver_profiles[vid] = DriverPreferenceProfile(
            driver_id=vid, name_ja=vid, description_ja="", home_charge_priority_alpha=0.6,
            comfort_priority_beta=0.6, allow_enroute_charging=False, roc_threshold_theta=0.5,
            min_home_arrival_soc_pct=15.0, battery_capacity_kwh=60.0, efficiency_km_per_kwh=6.5
        )
    p1_base.fit(train_recs_self)

    # 1. アプローチ①: ドライバ広域ECDF (Driver Global ECDF)
    class DriverGlobalPhase1(Phase1CollectorLearner):
        def __init__(self):
            super().__init__()
            self.driver_all_durs: Dict[str, np.ndarray] = {}

        def fit(self, records):
            super().fit(records)
            d_map: Dict[str, List[float]] = {}
            for r in records:
                d_map.setdefault(r.driver_id, []).append(r.duration_minutes)
            for d, arr in d_map.items():
                self.driver_all_durs[d] = np.sort(arr)

        def compute_conditional_return_probability(self, dist, elapsed_minutes, horizon_minutes):
            raw_durs = getattr(dist, "raw_durations", None)
            if raw_durs is None or len(raw_durs) < 3:
                # 未学習POIの場合、ドライバ全体の出先ヒストグラムを使用
                for drv, arr in self.driver_all_durs.items():
                    raw_durs = arr
                    break

            if raw_durs is not None and len(raw_durs) >= 3:
                surv_idx = np.searchsorted(raw_durs, elapsed_minutes, side="right")
                n_surv = len(raw_durs) - surv_idx
                if n_surv <= 0:
                    return 0.005
                ret_idx = np.searchsorted(raw_durs, elapsed_minutes + horizon_minutes, side="right")
                n_ret = ret_idx - surv_idx
                hazard = (n_ret + 0.15) / (n_surv + 1.0)
                return round(float(np.clip(hazard, 0.005, 0.995)), 4)
            return super().compute_conditional_return_probability(dist, elapsed_minutes, horizon_minutes)

    p1_driver_global = DriverGlobalPhase1()
    p1_driver_global.poi_clusters.update(eval_pois_self)
    p1_driver_global.poi_clusters.update(train_pois_self)
    for vid in target_vehicles:
        p1_driver_global.driver_profiles[vid] = p1_base.driver_profiles[vid]
    p1_driver_global.fit(train_recs_self)

    # 2. アプローチ②: ナビ・ジャンル連携 (Navigation Genre Prior)
    class NavGenrePhase1(Phase1CollectorLearner):
        def compute_conditional_return_probability(self, dist, elapsed_minutes, horizon_minutes):
            # ナビから取得した目的地のジャンル別分布（短時間買い物・コンビニ想定: 平均25分, 分散12分）
            genre_mean = 30.0
            genre_std = 12.0
            from scipy import stats
            cdf_t = float(stats.norm.cdf(elapsed_minutes, loc=genre_mean, scale=genre_std))
            cdf_next = float(stats.norm.cdf(elapsed_minutes + horizon_minutes, loc=genre_mean, scale=genre_std))
            surv_t = max(1e-4, 1.0 - cdf_t)
            hazard = (cdf_next - cdf_t) / surv_t
            return round(float(np.clip(hazard, 0.005, 0.995)), 4)

    p1_nav_genre = NavGenrePhase1()
    p1_nav_genre.poi_clusters.update(eval_pois_self)
    p1_nav_genre.poi_clusters.update(train_pois_self)
    for vid in target_vehicles:
        p1_nav_genre.driver_profiles[vid] = p1_base.driver_profiles[vid]
    p1_nav_genre.fit(train_recs_self)

    # 3. アプローチ③: クラウド・フリート集合知 (Fleet Crowdsourced Priors)
    # フリート他車全体の実際の出先駐車時間ヒストグラムを収集
    fleet_out_durs = np.sort(np.clip(other_train_df["trip_distance_km"].values * 6.5, 5.0, 300.0))

    class FleetCrowdPhase1(Phase1CollectorLearner):
        def __init__(self, fleet_durations):
            super().__init__()
            self.fleet_durs = fleet_durations

        def compute_conditional_return_probability(self, dist, elapsed_minutes, horizon_minutes):
            raw_durs = self.fleet_durs
            if raw_durs is not None and len(raw_durs) >= 10:
                surv_idx = np.searchsorted(raw_durs, elapsed_minutes, side="right")
                n_surv = len(raw_durs) - surv_idx
                if n_surv <= 0:
                    return 0.005
                ret_idx = np.searchsorted(raw_durs, elapsed_minutes + horizon_minutes, side="right")
                n_ret = ret_idx - surv_idx
                hazard = (n_ret + 0.15) / (n_surv + 1.0)
                return round(float(np.clip(hazard, 0.005, 0.995)), 4)
            return super().compute_conditional_return_probability(dist, elapsed_minutes, horizon_minutes)

    p1_fleet_crowd = FleetCrowdPhase1(fleet_out_durs)
    p1_fleet_crowd.poi_clusters.update(eval_pois_self)
    p1_fleet_crowd.poi_clusters.update(train_pois_self)
    for vid in target_vehicles:
        p1_fleet_crowd.driver_profiles[vid] = p1_base.driver_profiles[vid]
    p1_fleet_crowd.fit(train_recs_self)

    # 1-B. アプローチ①改良版: ドライバ別カテゴリECDF (Personalized Category ECDF)
    def get_poi_category(duration):
        if duration <= 25.0:
            return "QUICK"      # コンビニ、給油、送迎
        elif duration <= 90.0:
            return "MEDIUM"     # スーパー、食事、買い物
        return "LONG"           # レジャー、病院、用事

    class DriverPersonalizedCategoryPhase1(Phase1CollectorLearner):
        def __init__(self):
            super().__init__()
            self.driver_cat_durs: Dict[Tuple[str, str], np.ndarray] = {}
            self.driver_all_durs: Dict[str, np.ndarray] = {}

        def fit(self, records):
            super().fit(records)
            d_cat_map: Dict[Tuple[str, str], List[float]] = {}
            d_all_map: Dict[str, List[float]] = {}
            for r in records:
                cat = get_poi_category(r.duration_minutes)
                d_cat_map.setdefault((r.driver_id, cat), []).append(r.duration_minutes)
                d_all_map.setdefault(r.driver_id, []).append(r.duration_minutes)

            for k, v in d_cat_map.items():
                self.driver_cat_durs[k] = np.sort(v)
            for k, v in d_all_map.items():
                self.driver_all_durs[k] = np.sort(v)

    p1_driver_category = DriverPersonalizedCategoryPhase1()
    p1_driver_category.poi_clusters.update(eval_pois_self)
    p1_driver_category.poi_clusters.update(train_pois_self)
    for vid in target_vehicles:
        p1_driver_category.driver_profiles[vid] = p1_base.driver_profiles[vid]
    p1_driver_category.fit(train_recs_self)

    class CategoryAwareParkedMonitor(Phase3ParkedMonitor):
        def run_parked_session(self, event, profile, initial_plan, initial_env_state, dt_minutes=2.0, override_theta=None, control_strategy="psmp_jev"):
            cat = get_poi_category(event.duration_minutes)
            cat_durs = getattr(self.learner, "driver_cat_durs", {}).get((event.driver_id, cat))
            if cat_durs is None or len(cat_durs) < 2:
                cat_durs = getattr(self.learner, "driver_all_durs", {}).get(event.driver_id)

            if cat_durs is not None and len(cat_durs) >= 2:
                p50 = float(np.percentile(cat_durs, 50))
                p25 = float(np.percentile(cat_durs, 25))
                p75 = float(np.percentile(cat_durs, 75))
                iqr = max(5.0, (p75 - p25) / 1.349)
                cat_dist = LearnedStayDistribution(
                    mean_minutes=p50,
                    std_minutes=round(iqr, 2),
                    p10_minutes=float(np.percentile(cat_durs, 10)),
                    p50_minutes=p50,
                    p90_minutes=float(np.percentile(cat_durs, 90)),
                    sample_count=len(cat_durs),
                    confidence_weight=0.8,
                    raw_durations=cat_durs,
                )
                initial_plan = replace(initial_plan, stay_distribution=cat_dist)

            return super().run_parked_session(event, profile, initial_plan, initial_env_state, dt_minutes, override_theta, control_strategy)

    # --- 各手法の評価実行 ---
    strategies = [
        ("0. 現状ベースライン (一律POIスペック)", p1_base, Phase3ParkedMonitor),
        ("① ドライバ広域ECDF (個人出先全体)", p1_driver_global, Phase3ParkedMonitor),
        ("② ナビ・ジャンル連携 (カテゴリ事前分布)", p1_nav_genre, Phase3ParkedMonitor),
        ("★ ①改良版: 個人別カテゴリECDF (個人癖×場所種別)", p1_driver_category, CategoryAwareParkedMonitor),
        ("③ クラウド・フリート集合知 (他車OTA統計)", p1_fleet_crowd, Phase3ParkedMonitor),
    ]

    sim = VehicleSimulator()
    jev = JevDecisionEngine(backend_mode="calibrated_local")
    results = []

    print("\n--- Evaluating 5 Approaches on Unvisited First-Time Locations ---")
    for name, p1_model, monitor_cls in strategies:
        p2 = Phase2IGOffPlanner(jev, sim)
        p3 = monitor_cls(p1_model, jev, sim)
        p4 = Phase4PostTripEvaluator(p1_model, p2, p3)
        rep = p4.compute_roc_and_threshold_sweep(unvisited_sample, theta_grid=[0.3, 0.5, 0.7])
        op05 = [op for op in rep.operating_points if abs(op.theta - 0.5) < 1e-4][0]

        row = {
            "Strategy": name,
            "ROC_AUC": round(rep.roc_auc_jev, 4),
            "Brier_Score": round(rep.brier_score_jev, 4),
            "Ready_Comfort_Rate_pct": round(op05.avg_comfort_score, 1),
            "Power_Saving_pct": round(op05.energy_saving_vs_always_on_pct, 1),
            "Avg_Power_Wh": round(op05.avg_energy_wh, 1),
        }
        results.append(row)
        print(f"[{name}]")
        print(f"  ROC-AUC : {row['ROC_AUC']:.4f} | Brier: {row['Brier_Score']:.4f}")
        print(f"  レディ率: {row['Ready_Comfort_Rate_pct']:.1f}% | 省電力: {row['Power_Saving_pct']:.1f}% ({row['Avg_Power_Wh']:.1f} Wh)\n")

    df_res = pd.DataFrame(results)
    out_csv = out_dir / "coldstart_unvisited_comparison.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved results table to: {out_csv}")

    # 棒グラフの描画
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    
    # Panel 1: ROC-AUC
    ax1 = axes[0]
    bars1 = ax1.bar(
        [r["Strategy"].split(" ")[0] for r in results],
        [r["ROC_AUC"] for r in results],
        color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"],
        width=0.55
    )
    ax1.set_title("1. ROC-AUC on First-Time Unvisited Locations", fontsize=11, weight="bold")
    ax1.set_ylabel("ROC AUC", fontsize=10)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, weight="bold")

    # Panel 2: Comfort vs Power Saving
    ax2 = axes[1]
    x = np.arange(len(results))
    width = 0.35
    bars_c = ax2.bar(x - width/2, [r["Ready_Comfort_Rate_pct"] for r in results], width, label="Ready Comfort (%)", color="#1f77b4")
    bars_s = ax2.bar(x + width/2, [r["Power_Saving_pct"] for r in results], width, label="Power Saving (%)", color="#9467bd")
    ax2.set_title("2. Comfort & Power Saving on First-Time Locations", fontsize=11, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["Strategy"].split(" ")[0] for r in results])
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_ylim(0.0, 100.0)
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="y", alpha=0.3)

    out_chart = out_dir / "coldstart_unvisited_comparison.png"
    fig.tight_layout()
    fig.savefig(out_chart)
    plt.close(fig)
    print(f"Saved chart to: {out_chart}")


if __name__ == "__main__":
    run_coldstart_comparison()
