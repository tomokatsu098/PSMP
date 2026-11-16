"""
Verification script for integrated cold-start strategy in core Phase 1 module.
"""
from psmp.phase1_collector_learner import Phase1CollectorLearner
from psmp.real_data_loader import RealTripDataLoader
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator
from psmp.config import DriverPreferenceProfile

def main():
    train_csv = "data/syn548_profile_v4_train.csv.gz"
    eval_csv = "data/syn548_profile_v4_eval.csv.gz"
    target_vehicles = ["A00001", "A00002", "B00001", "B00002"]

    loader = RealTripDataLoader(train_csv, eval_csv)
    train_recs, train_pois = loader.get_train_records(vehicle_ids=target_vehicles, max_train_days=14, out_of_home_only=True)
    eval_recs, eval_pois = loader.get_eval_records(vehicle_ids=target_vehicles, out_of_home_only=True)

    known_poi_ids = set(r.poi_id for r in train_recs)
    unvisited = [r for r in eval_recs if r.poi_id not in known_poi_ids][:50]
    print(f"Unvisited count: {len(unvisited)}")

    p1 = Phase1CollectorLearner()
    p1.poi_clusters.update(eval_pois)
    p1.poi_clusters.update(train_pois)
    for vid in target_vehicles:
        p1.driver_profiles[vid] = DriverPreferenceProfile(
            driver_id=vid, name_ja=vid, description_ja="", home_charge_priority_alpha=0.6,
            comfort_priority_beta=0.6, allow_enroute_charging=False, roc_threshold_theta=0.5,
            min_home_arrival_soc_pct=15.0, battery_capacity_kwh=60.0, efficiency_km_per_kwh=6.5
        )
    p1.fit(train_recs)

    jev = JevDecisionEngine(backend_mode="calibrated_local")
    sim = VehicleSimulator()
    p2 = Phase2IGOffPlanner(jev, sim)
    p3 = Phase3ParkedMonitor(p1, jev, sim)
    p4 = Phase4PostTripEvaluator(p1, p2, p3)

    # ナビゲーション連携による目的地カテゴリ（QUICK/MEDIUM/LONG）指定時の評価
    # （カーナビやカレンダーから目的地の滞在枠カテゴリが与えられたシナリオ）
    class NavCategoryParkedMonitor(Phase3ParkedMonitor):
        def run_parked_session(self, event, profile, initial_plan, initial_env_state, dt_minutes=2.0, override_theta=None, control_strategy="psmp_jev"):
            from psmp.config import classify_stay_duration_category
            from dataclasses import replace
            target_cat = classify_stay_duration_category(event.duration_minutes)
            _, cat_dist = self.learner.predict_stay_distribution(
                driver_id=event.driver_id,
                lat=event.lat,
                lon=event.lon,
                park_start_dt=event.park_start_dt,
                is_holiday=event.is_holiday,
                has_passengers=event.has_passengers,
                target_category=target_cat,
            )
            initial_plan = replace(initial_plan, stay_distribution=cat_dist)
            return super().run_parked_session(event, profile, initial_plan, initial_env_state, dt_minutes, override_theta, control_strategy)

    p3_nav = NavCategoryParkedMonitor(p1, jev, sim)
    p4_nav = Phase4PostTripEvaluator(p1, p2, p3_nav)
    rep_nav = p4_nav.compute_roc_and_threshold_sweep(unvisited, theta_grid=[0.5])
    op05_nav = rep_nav.operating_points[0]

    print("========================================================")
    print("Integrated Core Module with Navigation Category Context:")
    print(f"  ROC-AUC: {rep_nav.roc_auc_jev:.4f} (HANDOFF.md reported 0.8106)")
    print(f"  Brier: {rep_nav.brier_score_jev:.4f} (HANDOFF.md reported 0.0477)")
    print(f"  Ready Rate: {op05_nav.avg_comfort_score:.1f}% (HANDOFF.md reported 80.1%)")
    print(f"  Power Saving: {op05_nav.energy_saving_vs_always_on_pct:.1f}% ({op05_nav.avg_energy_wh:.1f} Wh)")
    print("========================================================")


if __name__ == "__main__":
    main()
