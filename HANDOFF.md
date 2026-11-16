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

## 3. okuya-x1c の Antigravity でのプロンプト例

okuya-x1c で Antigravity を開いた後、以下のプロンプトをそのまま送信することで、コンテキストを即座に再開できます。

```text
okuya-x1pro-ubuntu で行っていた PSMP プロジェクトの作業を okuya-x1c で継続します。
リポジトリ内の HANDOFF.md を確認してください。
直近では実データ（syn548_profile_v4_train/eval.csv.gz）を用いて、
未訪問先（Cold-start）における「①改良版：ドライバ別カテゴリECDF」が ROC-AUC 0.8106、レディ達成率 80.1% を達成したところです。
この成果を踏まえ、次のステップ（例: 車載EdgeJevへの本番モジュール統合、あるいはUIアプリとの連携）を進めてください。
```
