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

---

## 4. okuya-x1c での委託元要求分析・仕様策定・実装実績（2026-11-17）

### (1) 委託元要求事項とベースライン (`deep7.py`) の解析
- 委託元資料（`past_materials/電力マネジメント_技術概要.pptx`、`deep7.py`）を徹底調査。
- **背景と目的**: 30日放置での補機バッテリあがり防止、補機1サイズUP（2,000円/台）回避のための **暗電流5mA以上削減**。
- **目標値**: 長期駐車（3日以上）で **DeepSleep 7.0 h/台日 以上** かつ **ドア解錠遅延率 1.2% 以下**（年間1,000回中月1回まで）。
- **委託元の課題**: `deep7.py` は全体一律閾値・ガードなしのベースラインであり、「全車で達成」やマイコン適合、突発利用の分離が未達。

### (2) 自宅（HOME）と出先（OUT）の完全分別アーキテクチャ実装
- 出先の短時間・中時間滞在分布と、自宅の夜間睡眠（8〜16h）や長期放置（3日以上）が混在するとECDFが歪むため、2層構造に完全分離。
  - **`HOME_OVERNIGHT`**: <= 24h（夜間即時DeepSleep突入）
  - **`HOME_LONG`**: > 24h（3日以上長期放置ターゲット）
  - **`QUICK` / `MEDIUM` / `LONG`**: 出先駐車用
- `psmp/config.py`: `POIType.HOME` 追加、`classify_stay_duration_category` の `is_home` 対応。
- `psmp/phase1_collector_learner.py`: `driver_home_durations` / `driver_out_durations` 専用プールの新設、自宅フォールバック統合。
- `psmp/real_data_loader.py`: `prev_loc == "自宅"` のタグ付け対応。
- `tests/test_coldstart.py`: 自宅と出先の分別ユニットテストを新設し、**全5テスト全て PASSED**。

### (3) 電力マネジメント仕様策定
- **ユースケース体系＆星取り表**: 委託元（暗電流37mAカットのみ）vs PSMP（高圧連携、Re-sleep、SoCガード、マイコン超軽量化）の優位性を整理。
- **省電力モード状態遷移表**: S0(Drive), S1(Pre-con), S2(Normal Sleep), S3(DeepSleep), S4(Emergency Cut) の遷移条件と安全弁を定義。
- **入力データ構造**: 1,000台、548日、Aグループ（103台: 高難度通勤型）vs Bグループ（897台: 日常生活圏型）の理由と内訳を整理。
- **ユーザの嬉しさ評価指標**: ワンアクション解錠率（$\ge 98.8\%$）、30日始動率（$100\%$）、無充電帰還率（$100\%$）、快適性スコア（$\ge 80$点）。
- **UI/UX役割分担**: 車載AAOS（確認・最小限） vs スマホ/自宅PC（リモート把握・安心スイッチ・詳細設定）。
- 詳細仕様書: [`docs/POWER_MANAGEMENT_SPEC.md`](docs/POWER_MANAGEMENT_SPEC.md) を新設。

---

## 5. 今夜 okuya-x1pro-ubuntu 側での作業タスク（ToDoリスト）

夜間、母艦 `okuya-x1pro-ubuntu` で作業を再開する際は、以下のステップで進めてください。

### (1) 環境同期と健全性確認
```bash
cd ~/develop/PSMP  # または ~/git/PSMP
git pull origin main
source .venv/bin/activate
pytest -v  # 5件すべてパスすることを確認
```

### (2) 委託元クライテリア達成に向けた実装＆評価タスク
1. **長期駐車（3日以上 / `HOME_LONG`）専用ロジックの評価スクリプト作成**:
   - `data/syn548_profile_v4_eval.csv.gz`（または全1,000台データ）を対象に、`HOME_LONG` セッションで「DeepSleep 7.0 h/台日 以上 かつ 解錠遅延 1.2% 以下」が**全車で達成できているか**を検証する評価コードを作成。
2. **突発利用（段階2）の定量的定義と理論上限値の導出**:
   - 「過去実績のない時間帯・遠隔地」を突発利用として定義。
   - 突発利用を除外した際、遅延0件でDeepSleepが理論上何時間まで伸ばせるかを算出。
3. **委託元提出用検討書（①〜⑤）の下書き作成**:
   - `docs/POWER_MANAGEMENT_SPEC.md` の内容をベースに、委託元への提出成果物（入力成形、モデル選定、判断ロジック、個人適応、システム方式）の構成案を作成。

