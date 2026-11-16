# PSMP (Power-Saving Management for Parked Vehicles)

テスラ等のEVやプリウスPHEVなどが出先の駐車場に駐車した際、車載機器を適切に省電力モードへ移行させ、ドライバが戻ってくる時間を予測して事前に必要な機器を復帰・レディ状態にし、スムーズな発進と電費（自宅無充電帰還）のトレードオフを最適化する省電力マネジメント機構のプロトタイプおよび精度評価フレームワークです。

## 4フェーズ (Phase 1 〜 Phase 4) アーキテクチャ

1. **Phase 1: 学習すべき情報を収集するフェーズ (`psmp/phase1_collector_learner.py`)**
   - ドライバIDごとの駐車開始タイミング (`IG-OFF`) と発進タイミング (`IG-ON`)、座標（POIクラスタ）、日時（日付・時間・曜日・平日/休日）、同乗者有無を収集し、条件付き滞在時間確率分布と個人嗜好プロファイルを学習・保持します。
2. **Phase 2: 駐車してIG-OFFしてから省電力マネジメントを実施するフェーズ (`psmp/phase2_ig_off_planner.py`)**
   - `IG-OFF` 直後に自宅までの帰還必要電力量と現在SoCから駐車中の余剰電力バジェットを算出し、**Jev ローカル判断特化型AI (`system_one`: `choice` / `score`)** によって初期ポリシーモードと車載4機器（①空調 HVAC、②IVI/ナビ/通信、③駆動用バッテリー温調、④監視カメラ Sentry）の初期省電力状態を決定します。
3. **Phase 3: 駐車中、ドライバが戻ってくるまで監視・待機・省電力モード調整するフェーズ (`psmp/phase3_parked_monitor.py`)**
   - 駐車中に環境・経過時間・SoCを監視し、**Jev ローカル判断特化型AI (`noul` / `score`)** で各機器のリードタイム内の帰還確率 $P(\text{wake})$ を1パス推論します。
   - **ROC動作点閾値 $\theta$** に基づく段階的レディ復帰に加え、予定遅延時の自動再スリープ (`Re-sleep Guard`)、低SoC時の自宅帰還優先カットオフ (`Range Guard Override`)、猛暑・厳冬・同乗者あり時の優先復帰を動的に実行します。
4. **Phase 4: 帰宅後、効果を可視化し、かつ、次回に向けて再調整できるUIフェーズ (`psmp/phase4_post_trip_evaluator.py` & `app.py`)**
   - 帰宅後に省電力削減量 (Wh)、入車時快適度・レディ達成率、無充電帰宅の成否を可視化します。
   - **ROC感度特性曲線 (Sensitivity vs FPR)** 上で動作点閾値 $\theta$ や個人嗜好バランス（自宅充電こだわり度 $\alpha$ vs 車内快適性優先度 $\beta$）を調整し、Phase 1 プロファイルへフィードバックします。

## 実行方法

```bash
# 1. 単体・統合テストの実行
.venv/bin/pytest -v

# 2. Phase 1〜4 一括シミュレーション＆ROC感度評価レポート・グラフ生成
.venv/bin/python run_evaluation.py

# 3. Phase 4 インタラクティブ可視化＆ROC再調整ダッシュボード (Streamlit) の起動
.venv/bin/streamlit run app.py
```
