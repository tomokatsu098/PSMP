# PSMP (Parking State Management & Prediction) 開発引き継ぎドキュメント

本ドキュメントは、**okuya-x1pro-ubuntu** から **okuya-x1c** への環境移行および Antigravity での作業継続のための引き継ぎサマリーです。

---

## 1. okuya-x1c での環境セットアップ手順

### (1) リポジトリのクローン
```bash
git clone git@github.com:tomokatsu098/PSMP.git
cd PSMP
```

### (2) Python 仮想環境の作成と依存関係のインストール
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
※ `requirements.txt` に記載されている主要ライブラリ: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`

### (3) データファイルの確認
- `data/syn548_profile_v4_train.csv.gz` (約21MB)
- `data/syn548_profile_v4_eval.csv.gz` (約11MB)
※ pandas が透過的に `.csv.gz` を直接読み込むため、事前の手動解凍は不要です。

### (4) 動作検証
```bash
# 初訪問（Cold-start）評価スクリプトのテスト実行
python evaluate_coldstart_strategies.py
```

---

## 2. これまでのタスクと進捗サマリー

### (1) 入力データ概要
- 1,000台のEV/HEVによる548日（約1.5年分）の実走行トリップログ。
- トリップから「駐車セッション（IG-OFF $\to$ 次のIG-ON）」を抽出（`psmp/real_data_loader.py`）。
- 暦日分割済み：
  - **学習用**: `syn548_profile_v4_train.csv.gz`（2025-04-02 〜 2026-03-31、約99.8万行）
  - **評価用**: `syn548_profile_v4_eval.csv.gz`（2026-04-01 〜 2026-10-01、約49.9万行）

### (2) 主な改修と精度改善の経緯
1. **EdgeJevエンジンの精度低迷原因の解明**:
   - 実データの出先滞在時間は「5〜15分（コンビニ・送迎等）」「30〜60分（食事・買物）」「半日・放置」が混在する**多峰型（Multimodal）分布**。
   - 従来の単一正規分布モデル $\mathcal{N}(\mu, \sigma^2)$ では確率が潰れてしまうため、**ECDF（経験的累積分布／カプラン・マイヤー生存関数）によるハザード率算出** へ移行。
2. **学習ステップ期間別検証（案3）**:
   - 7日、14日、30日、60日、90日、180日、365日の期間長で評価。
   - 365日学習で EdgeJev の ROC-AUC は 0.6969（約0.70）、Brierスコア 0.1051、レディ率 77.7% に到達（`evaluate_learning_steps.py`）。
3. **初めて行く場所（未学習POI/Cold-start）の解決**:
   - 未訪問先（50セッション）における予測手法を比較。
   - **「①改良版：ドライバ別カテゴリECDF（個人癖 × 場所種別）」** が **ROC-AUC 0.8106、Brier 0.0477、レディ率 80.1%** を叩き出し、クラウド他車集合知（AUC 0.8028）を上回る最高性能を完全オフラインで達成（`evaluate_coldstart_strategies.py`）。

---

## 3. okuya-x1c での作業実績と完了ステータス（2026-11-16）

### (1) 本番コアモジュールへの統合完了
- `psmp/config.py`: 滞在時間カテゴリ分類関数 `classify_stay_duration_category`（QUICK/MEDIUM/LONG）および POI デフォルトカテゴリ判定 `get_poi_default_category` を追加。
- `psmp/phase1_collector_learner.py`:
  - `Phase1CollectorLearner.fit()` でドライバごとの滞在カテゴリ別ECDF `driver_category_durations` および全出先滞在時間 `driver_global_durations` を自動学習・保持。
  - `predict_stay_distribution()` において、未訪問先（Cold-start）や学習データ不足時に、目的地種別/カーナビジャンルに応じた「ドライバ別カテゴリECDF」を自動生成するフォールバックロジックを統合。
- 実データ（`syn548_profile_v4`）の未訪問50セッションによる再検証（`verify_integrated_coldstart.py`）:
  - **ROC-AUC: 0.8106**（従来ベースライン 0.4959）
  - **Brier スコア: 0.0477**（従来 0.1243）
  - **レディ達成率: 80.1%**、**省電力率: 47.8%**（平均消費 1782.9 Wh）
  - 完全オフライン（他車通信不要）での高精度制御がコアモジュール単体で再現されることを確認。

### (2) Streamlit UI アプリ (`app.py`) の拡張
- **Tab 3**: 学習済み分布テーブルに加え、ドライバごとの滞在カテゴリ（QUICK/MEDIUM/LONG）の中央値・IQR・最小最大スパンを表示するインスペクタを追加。
- **Tab 4 (新規)**: 「❄️ Cold-Start: 未訪問先 (初訪問POI) ＆ ナビカテゴリ連携」タブを追加。
  - カーナビ目的地ジャンル（QUICK/MEDIUM/LONG）、実滞在時間、外気温、同乗者の設定から、従来 baseline（一律スペック）vs ドライバ別カテゴリECDF の確率推移・室温推移・消費電力をリアルタイム比較可能。
  - 1,000台実データにおける5方式ベンチマーク比較表を提示。

### (3) テストスイート拡充
- `pyproject.toml`: setuptools のパッケージ探索設定を整備し、`pip install -e .` による editable インストールに対応。
- `tests/test_coldstart.py`: 未訪問POIにおけるドライバ別カテゴリECDFの自動適用、生存関数ハザード計算、Phase 2/Phase 3 連携を検証するユニットテストを新設。
- 全テスト（`pytest` 4件）が全て成功（PASSED）することを確認。

