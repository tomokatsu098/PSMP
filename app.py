"""
Phase 4 UI: PSMP Post-Trip Visualization & EdgeJev vs LightGBM ROC Sensitivity Tuning Dashboard (`app.py`)
４．帰宅後、効果を可視化し、かつ、次回に向けて再調整できるUIフェーズ
  - Tab 1: 帰宅後サマリー＆時系列可視化（EdgeJev vs LightGBM の確率・室温・消費電力の直接比較）
  - Tab 2: EdgeJev vs LightGBM の ROC感度特性曲線・Brierスコア・パレートフロント・特徴量重要度の比較＆閾値 θ 再調整UI
  - Tab 3: Phase 1 学習データ＆ドライバ×座標POI別滞在時間分布インスペクタ
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from psmp.config import DEFAULT_DRIVER_PROFILES, DEFAULT_POI_CLUSTERS
from psmp.models.jev_engine import JevDecisionEngine
from psmp.models.lgbm_engine import LightGBMDecisionEngine
from psmp.models.vehicle_devices import VehicleSimulator
from psmp.phase1_collector_learner import Phase1CollectorLearner
from psmp.phase2_ig_off_planner import Phase2IGOffPlanner
from psmp.phase3_parked_monitor import Phase3ParkedMonitor
from psmp.phase4_post_trip_evaluator import Phase4PostTripEvaluator


st.set_page_config(
    page_title="PSMP: EdgeJev vs LightGBM Power-Saving Management",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def init_psmp_system():
    phase1 = Phase1CollectorLearner()
    all_records = phase1.generate_synthetic_history(num_sessions_per_driver=100, seed=42, include_edge_cases=True)
    train_records = [r for idx, r in enumerate(all_records) if (idx % 100) < 75]
    test_records = [r for idx, r in enumerate(all_records) if (idx % 100) >= 75]
    phase1.fit(train_records)

    lgbm_engine = LightGBMDecisionEngine(random_state=42)
    lgbm_engine.fit(train_records, phase1)

    jev_engine = JevDecisionEngine(backend_mode="calibrated_local")
    vehicle_sim = VehicleSimulator()
    phase2 = Phase2IGOffPlanner(jev_engine, vehicle_sim)
    phase3 = Phase3ParkedMonitor(phase1, jev_engine, vehicle_sim, lgbm_engine=lgbm_engine)
    phase4 = Phase4PostTripEvaluator(phase1, phase2, phase3)
    return phase1, phase2, phase3, phase4, lgbm_engine, train_records, test_records


def main():
    phase1, phase2, phase3, phase4, lgbm_engine, train_records, test_records = init_psmp_system()

    st.title("🚗 PSMP: Power-Saving Management for Parked Vehicles")
    st.caption(
        "EdgeJev ローカル判断特化型AI (`edgejev` System One) vs LightGBM (`lightgbm` GBDT) の直接比較 ＆ ROC感度特性チューニング"
    )

    # --- Sidebar: ドライバ選択 ＆ 次回に向けた再調整コントロール ---
    st.sidebar.header("⚙️ ドライバ設定 & ROC感度再調整")
    driver_options = list(phase1.driver_profiles.keys())
    selected_driver_id = st.sidebar.selectbox(
        "対象ドライバIDを選択",
        driver_options,
        format_func=lambda d: f"{d}: {phase1.driver_profiles[d].name_ja}",
    )
    prof = phase1.driver_profiles[selected_driver_id]
    st.sidebar.info(prof.description_ja)

    st.sidebar.subheader("🤖 比較・主表示モデルの選択")
    primary_model = st.sidebar.radio(
        "帰宅後KPIカードの主表示モデル",
        ["psmp_jev", "psmp_lightgbm"],
        format_func=lambda m: "EdgeJev (ローカル判断特化型AI)" if m == "psmp_jev" else "LightGBM (GBDT 決定木モデル)",
    )

    st.sidebar.subheader("🎯 ROC動作点 (閾値 θ) 調整")
    tuned_theta = st.sidebar.slider(
        "復帰トリガー閾値 θ (低い=高感度/快適優先, 高い=省電力/空振り抑制)",
        min_value=0.10,
        max_value=0.90,
        value=float(prof.roc_threshold_theta),
        step=0.05,
    )

    st.sidebar.subheader("🔋 個人のバッテリー・快適性嗜好")
    tuned_alpha = st.sidebar.slider(
        "自宅無充電帰還こだわり度 α (Range Priority)",
        min_value=0.0,
        max_value=1.0,
        value=float(prof.home_charge_priority_alpha),
        step=0.05,
    )
    tuned_beta = st.sidebar.slider(
        "車内快適性優先度 β (Comfort Priority)",
        min_value=0.0,
        max_value=1.0,
        value=float(prof.comfort_priority_beta),
        step=0.05,
    )
    tuned_enroute = st.sidebar.checkbox(
        "快適性のためなら帰路での途中充電をやむなしとする",
        value=bool(prof.allow_enroute_charging),
    )

    if st.sidebar.button("💾 次回走行へ向けてPhase 1プロファイルへ保存", type="primary"):
        phase1.update_driver_preference(
            selected_driver_id,
            new_theta=tuned_theta,
            new_alpha_range=tuned_alpha,
            new_beta_comfort=tuned_beta,
            allow_enroute_charging=tuned_enroute,
        )
        st.sidebar.success(
            f"Phase 1 プロファイル更新完了！ (θ={tuned_theta:.2f}, α={tuned_alpha:.2f}, β={tuned_beta:.2f})"
        )

    prof.roc_threshold_theta = tuned_theta
    prof.home_charge_priority_alpha = tuned_alpha
    prof.comfort_priority_beta = tuned_beta
    prof.allow_enroute_charging = tuned_enroute

    tab1, tab2, tab3 = st.tabs([
        "📊 Phase 4: 帰宅後の効果可視化 (EdgeJev vs LightGBM 時系列比較)",
        "📈 EdgeJev vs LightGBM: ROC感度特性曲線 ＆ パレートトレードオフ比較",
        "🗂️ Phase 1: 学習済み滞在時間分布 ＆ LightGBM特徴量重要度",
    ])

    # =========================================================================
    # Tab 1: 帰宅後の効果可視化 (EdgeJev vs LightGBM 時系列比較)
    # =========================================================================
    with tab1:
        drv_events = [e for e in test_records if e.driver_id == selected_driver_id]
        col_sel1, col_sel2 = st.columns([2, 2])
        with col_sel1:
            selected_ev = st.selectbox(
                "振り返る駐車セッション（出先駐車場）を選択",
                drv_events,
                format_func=lambda e: (
                    f"{e.session_id} | {DEFAULT_POI_CLUSTERS[e.poi_id].name_ja} | "
                    f"{e.season} ({e.ambient_temp_c}°C) | 実滞在 {e.duration_minutes}分 | "
                    f"同乗者:{'あり' if e.has_passengers else 'なし'} | [{e.anomaly_type}]"
                ),
            )
        comp = phase4.evaluate_single_trip_comparison(selected_ev, custom_theta=tuned_theta)
        res_jev = comp["psmp_jev"]
        res_lgbm = comp["psmp_lightgbm"]
        res_ao = comp["always_on"]
        res_np = comp["no_precon"]
        res_active = comp[primary_model]
        other_res = res_lgbm if primary_model == "psmp_jev" else res_jev
        other_label = "LightGBM" if primary_model == "psmp_jev" else "EdgeJev"

        with col_sel2:
            lgbm_est_stay = lgbm_engine.predict_stay_duration(selected_ev)
            st.markdown(
                f"**Phase 2 (IG-OFF初期判定)**: `{res_jev.initial_policy_mode.upper()}`  \n"
                f"**Phase 1 予想平均**: `{res_jev.expected_mean_minutes:.1f}分` | **LightGBM回帰予測**: `{lgbm_est_stay:.1f}分` | **実滞在**: `{res_jev.actual_duration_minutes:.1f}分`  \n"
                f"**駐車開始時SoC**: `{res_jev.initial_soc_pct:.1f}%` / **自宅までの距離**: `{selected_ev.distance_to_home_km:.1f} km`"
            )

        saving_wh = res_ao.parking_energy_consumed_wh - res_active.parking_energy_consumed_wh
        saving_pct = (saving_wh / max(1.0, res_ao.parking_energy_consumed_wh)) * 100.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"🔋 駐車中の消費電力量 ({'EdgeJev' if primary_model=='psmp_jev' else 'LightGBM'})",
            f"{res_active.parking_energy_consumed_wh:.0f} Wh",
            delta=f"{other_label}: {other_res.parking_energy_consumed_wh:.0f} Wh (常時比 -{saving_pct:.1f}%)",
            delta_color="normal",
        )
        m2.metric(
            "🌡️ 入車時の車内温度",
            f"{res_active.departure_cabin_temp_c:.1f} °C",
            delta=f"目標 {res_active.target_cabin_temp_c:.1f}°C ({other_label}: {other_res.departure_cabin_temp_c:.1f}°C)",
            delta_color="off",
        )
        m3.metric(
            "✨ 入車時トータル快適・レディ度",
            f"{res_active.overall_ready_score:.1f} / 100",
            delta=f"{res_active.overall_ready_score - other_res.overall_ready_score:+.1f} pt vs {other_label} ({other_res.overall_ready_score:.1f})",
        )
        m4.metric(
            "🏠 自宅無充電帰還の成否",
            "✅ 無充電で帰宅可能" if res_active.can_return_home_without_charging else "⚡ 帰路で途中充電が必要",
            delta=f"帰宅時予想SoC: {res_active.estimated_home_arrival_soc_pct:.1f}%",
        )

        # 時系列インタラクティブチャート (EdgeJev vs LightGBM 並列比較)
        logs_jev = res_jev.timeline_logs
        logs_lgbm = res_lgbm.timeline_logs
        ts = [l.elapsed_minutes for l in logs_jev]
        fig_tl = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                "① EdgeJev (`noul`) vs LightGBM (`LGBMClassifier`): 事前復帰確率 P(wake_hvac) と ROC閾値 θ",
                "② 車内温度 (Cabin Temp) の推移比較: EdgeJev vs LightGBM",
                "③ 車載4機器の合計消費電力 (W) 推移比較: EdgeJev vs LightGBM",
            ),
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.jev_wake_hvac_prob for l in logs_jev], name="EdgeJev P(wake_hvac)", line=dict(color="#1f77b4", width=3)),
            row=1, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.lgbm_wake_hvac_prob for l in logs_jev], name="LightGBM P(wake_hvac)", line=dict(color="#2ca02c", width=2.5, dash="dashdot")),
            row=1, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.stat_hazard_prob for l in logs_jev], name="Coarse Stat Baseline", line=dict(color="#ff7f0e", width=1.8, dash="dash")),
            row=1, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.effective_threshold_theta for l in logs_jev], name="Effective ROC Threshold θ", line=dict(color="#d62728", width=2, dash="dot")),
            row=1, col=1,
        )

        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.cabin_temp_c for l in logs_jev], name="EdgeJev 車内温度 (°C)", line=dict(color="#1f77b4", width=3)),
            row=2, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[l.cabin_temp_c for l in logs_lgbm], name="LightGBM 車内温度 (°C)", line=dict(color="#2ca02c", width=2.5, dash="dashdot")),
            row=2, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(x=ts, y=[res_jev.target_cabin_temp_c] * len(ts), name="目標快適温度 (°C)", line=dict(color="#d62728", width=2, dash="dash")),
            row=2, col=1,
        )

        fig_tl.add_trace(
            go.Scatter(
                x=ts,
                y=[l.step_power_w for l in logs_jev],
                name="EdgeJev 合計消費電力 (W)",
                line=dict(color="#1f77b4", width=2.5, shape="hv"),
                fill="tozeroy",
            ),
            row=3, col=1,
        )
        fig_tl.add_trace(
            go.Scatter(
                x=ts,
                y=[l.step_power_w for l in logs_lgbm],
                name="LightGBM 合計消費電力 (W)",
                line=dict(color="#2ca02c", width=2.2, shape="hv", dash="dashdot"),
            ),
            row=3, col=1,
        )
        fig_tl.update_layout(height=680, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_tl, use_container_width=True)

    # =========================================================================
    # Tab 2: EdgeJev vs LightGBM ROC感度特性曲線 ＆ 閾値 θ トレードオフ比較
    # =========================================================================
    with tab2:
        st.subheader("📈 EdgeJev vs LightGBM: ROC感度特性曲線 (Sensitivity vs FPR) とパレートフロント比較")
        roc_report = phase4.compute_roc_and_threshold_sweep(test_records)

        k1, k2, k3 = st.columns(3)
        k1.metric("EdgeJev ROC AUC (Brier Loss)", f"{roc_report.roc_auc_jev:.4f}", f"Brier: {roc_report.brier_score_jev:.4f} (確率較正誤差)")
        k2.metric("LightGBM ROC AUC (Brier Loss)", f"{roc_report.roc_auc_lgbm:.4f}", f"Brier: {roc_report.brier_score_lgbm:.4f}")
        k3.metric("Statistical Baseline ROC AUC", f"{roc_report.roc_auc_stat_baseline:.4f}", f"EdgeJev差: {roc_report.roc_auc_jev - roc_report.roc_auc_stat_baseline:+.4f}")

        c1, c2 = st.columns(2)
        with c1:
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=roc_report.fpr_curve_jev,
                    y=roc_report.tpr_curve_jev,
                    mode="lines",
                    name=f"EdgeJev System-1 AI (AUC = {roc_report.roc_auc_jev:.4f})",
                    line=dict(color="#1f77b4", width=3.5),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=roc_report.fpr_curve_lgbm,
                    y=roc_report.tpr_curve_lgbm,
                    mode="lines",
                    name=f"LightGBM GBDT (AUC = {roc_report.roc_auc_lgbm:.4f})",
                    line=dict(color="#2ca02c", width=3.0, dash="dashdot"),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=roc_report.fpr_curve_stat,
                    y=roc_report.tpr_curve_stat,
                    mode="lines",
                    name=f"Statistical Timer Baseline (AUC = {roc_report.roc_auc_stat_baseline:.4f})",
                    line=dict(color="#ff7f0e", width=2.2, dash="dash"),
                )
            )
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="gray", dash="dot"))
            )

            nearest_jev = min(roc_report.operating_points, key=lambda op: abs(op.theta - tuned_theta))
            nearest_lgbm = min(roc_report.operating_points_lgbm, key=lambda op: abs(op.theta - tuned_theta))
            fig_roc.add_trace(
                go.Scatter(
                    x=[nearest_jev.fpr_false_alarm],
                    y=[nearest_jev.tpr_sensitivity],
                    mode="markers",
                    name=f"EdgeJev 動作点 (θ={nearest_jev.theta:.2f})",
                    marker=dict(color="#d62728", size=14, symbol="star"),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[nearest_lgbm.fpr_false_alarm],
                    y=[nearest_lgbm.tpr_sensitivity],
                    mode="markers",
                    name=f"LightGBM 動作点 (θ={nearest_lgbm.theta:.2f})",
                    marker=dict(color="#2ca02c", size=13, symbol="diamond"),
                )
            )
            fig_roc.update_layout(
                title="ROC Curve: EdgeJev vs LightGBM vs Baseline",
                xaxis_title="False Positive Rate (FPR: 空振りによる無駄稼働率)",
                yaxis_title="True Positive Rate (TPR / Sensitivity: 入車時レディ感度)",
                height=460,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with c2:
            op_jev_df = pd.DataFrame([vars(op) for op in roc_report.operating_points])
            op_lgbm_df = pd.DataFrame([vars(op) for op in roc_report.operating_points_lgbm])
            fig_tr = go.Figure()
            fig_tr.add_trace(
                go.Scatter(x=op_jev_df["avg_energy_wh"], y=op_jev_df["avg_comfort_score"], mode="lines+markers", name="EdgeJev パレート曲線 (Sweep θ)", line=dict(color="#1f77b4", width=3))
            )
            fig_tr.add_trace(
                go.Scatter(x=op_lgbm_df["avg_energy_wh"], y=op_lgbm_df["avg_comfort_score"], mode="lines+markers", name="LightGBM パレート曲線 (Sweep θ)", line=dict(color="#2ca02c", width=2.5, dash="dashdot"))
            )
            fig_tr.update_layout(
                title="パレートトレードオフ: 平均消費電力 (Wh) vs 入車時快適・レディスコア",
                xaxis_title="平均消費電力量 (Wh) [左に行くほど省電力]",
                yaxis_title="入車時快適・レディスコア (0-100) [上に行くほど快適]",
                height=460,
            )
            st.plotly_chart(fig_tr, use_container_width=True)

        st.subheader("🏆 5方式の総合ベンチマーク比較表 (EdgeJev vs LightGBM vs Baselines)")
        st.dataframe(roc_report.strategy_comparison_df, use_container_width=True)

    # =========================================================================
    # Tab 3: Phase 1 学習済み滞在時間分布 ＆ LightGBM 特徴量重要度
    # =========================================================================
    with tab3:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            st.subheader("🌲 LightGBM 特徴量重要度 (%)")
            if lgbm_engine.fit_summary is not None:
                imp_df = pd.DataFrame(
                    list(lgbm_engine.fit_summary.feature_importances.items()),
                    columns=["Feature", "Importance_Pct"],
                )
                st.dataframe(imp_df, use_container_width=True)
        with col_f2:
            st.subheader("🗂️ Phase 1: ドライバ × 座標(POI) × 曜日 × 時間帯 × 同乗者別の学習済み分布")
            learned_rows = []
            for key, dist in phase1.learned_stats.items():
                drv_id, poi_id, is_hol, tb, has_pass = key
                learned_rows.append(
                    {
                        "Driver_ID": drv_id,
                        "POI_ID": poi_id,
                        "POI_Name": DEFAULT_POI_CLUSTERS[poi_id].name_ja,
                        "Is_Holiday": "休日" if is_hol else "平日",
                        "Time_Bucket": tb,
                        "Has_Passengers": "同乗者あり" if has_pass else "単独",
                        "Mean_Stay_Min": dist.mean_minutes,
                        "Std_Min": dist.std_minutes,
                        "Sample_Count": dist.sample_count,
                    }
                )
            st.dataframe(pd.DataFrame(learned_rows), use_container_width=True)


if __name__ == "__main__":
    main()
