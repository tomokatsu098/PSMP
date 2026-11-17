# PSMP (Parking State Management & Prediction) 開発引き継ぎドキュメント

本ドキュメントは、**okuya-x1pro-ubuntu** と **okuya-x1c** の間での環境移行および Antigravity での作業継続のための引き継ぎサマリーです。

---

## 1. okuya-x1c / okuya-x1pro-ubuntu での環境セットアップ手順

### (1) リポジトリのクローン・同期
```bash
git clone git@github.com:tomokatsu098/PSMP.git
cd PSMP
git pull origin main
```

### (2) Python 仮想環境の作成と依存関係のインストール
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
※ 主要ライブラリ: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`

### (3) データファイルの確認
- `data/syn548_profile_v4_train.csv.gz` (約21MB)
- `data/syn548_profile_v4_eval.csv.gz` (約11MB)
※ pandas が透過的に `.csv.gz` を直接読み込むため、事前の手動解凍は不要です。

### (4) 動作検証
```bash
pytest -v  # 全5件のユニットテストが全てPASSEDすることを確認
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
   - 実データの出先滞在時間は「5〜15分（コンビニ等）」「30〜60分（食事・買物）」「半日・放置」が混在する**多峰型（Multimodal）分布**。
   - 従来の単一正規分布モデル $\mathcal{N}(\mu, \sigma^2)$ では確率が潰れてしまうため、**ECDF（経験的累積分布／カプラン・マイヤー生存関数）によるハザード率算出** へ移行。
2. **学習ステップ期間別検証（案3）**:
   - 7日、14日、30日、60日、90日、180日、365日の期間長で評価。
   - 365日学習で EdgeJev の ROC-AUC は 0.6969（約0.70）、Brierスコア 0.1051、レディ率 77.7% に到達（`evaluate_learning_steps.py`）。
3. **初めて行く場所（未学習POI/Cold-start）の解決**:
   - 未訪問先（50セッション）における予測手法を比較。
   - **「①改良版：ドライバ別カテゴリECDF（個人癖 × 場所種別）」** が **ROC-AUC 0.8106、Brier 0.0477、レディ率 80.1%** を叩き出し、クラウド他車集合知（AUC 0.8028）を上回る最高性能を完全オフラインで達成（`evaluate_coldstart_strategies.py`）。

---

## 3. okuya-x1c での作業実績（2026-11-16）

### (1) 本番コアモジュールへの統合完了
- `psmp/config.py`: 滞在時間カテゴリ分類関数 `classify_stay_duration_category`（QUICK/MEDIUM/LONG）および POI デフォルトカテゴリ判定 `get_poi_default_category` を追加。
- `psmp/phase1_collector_learner.py`:
  - `Phase1CollectorLearner.fit()` でドライバごとの滞在カテゴリ別ECDF `driver_category_durations` および全出先滞在時間 `driver_global_durations` を自動学習・保持。
  - `predict_stay_distribution()` において、未訪問先（Cold-start）や学習データ不足時に、目的地種別/カーナビジャンルに応じた「ドライバ別カテゴリECDF」を自動生成するフォールバックロジックを統合。

### (2) Streamlit UI アプリ (`app.py`) の拡張
- **Tab 3**: 学習済み分布テーブルに加え、ドライバごとの滞在カテゴリ（QUICK/MEDIUM/LONG）の中央値・IQR・最小最大スパンを表示するインスペクタを追加。
- **Tab 4 (新規)**: 「❄️ Cold-Start: 未訪問先 (初訪問POI) ＆ ナビカテゴリ連携」タブを追加。

---

## 4. okuya-x1c での委託元要求分析・仕様策定・シミュレーション実績（2026-11-17）

### (1) 委託元要求事項とベースライン (`deep7.py`) の解析
- 委託元資料（`past_materials/電力マネジメント_技術概要.pptx`、`deep7.py`）を徹底調査。
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

### (3) 電力マネジメント仕様策定と全5項目検討書ドラフト完成
- **詳細仕様書**: [`docs/POWER_MANAGEMENT_SPEC.md`](docs/POWER_MANAGEMENT_SPEC.md)
  - ユースケース体系＆星取り表（委託元 vs PSMP）
  - 省電力モード状態遷移表＆Mermaid図（S0〜S4、Re-sleep・SoC安全弁）
  - 入力データ仕様（1,000台、548日、A/Bグループの理由と内訳）
  - ユーザの嬉しさ評価指標（ワンアクション解錠率 $\ge 98.8\%$、30日始動率 $100\%$ 等）
  - 車載AAOS と スマホ/クラウドPC のUI/UX機能分担（ゼロクリック自動判断優先）
- **委託元提出用技術検討書（全5項目網羅）**: [`docs/PROPOSAL_TECHNICAL_REPORTS_DRAFT.md`](docs/PROPOSAL_TECHNICAL_REPORTS_DRAFT.md)
  - ① 入力成形検討書（102次元 $\to$ 14次元へ圧縮し、AUC 0.8106、Brier 0.0474達成）
  - ② AIモデル選定検討書（C-DCマイコン向けにROM 8.5KB、推論3μsのEdgeJev/ECDF選定）
  - ③ 判断ロジック最適化検討書（段階1: 長期駐車 18.0h/日達成、段階2: 全車0件遅延・理論値 20.5h/日達成）
  - ④ 個人適応最適化検討書（個人適応により段階2達成率が 24.3% $\to$ 100%へ飛躍）
  - ⑤ システム方式検討書（クラウド学習＋車載EdgeJev推論のハイブリッド分散型DA表策定）

### (4) 長期駐車＆突発利用 全1,000台シミュレーション完了 (`evaluate_long_parking_and_spontaneous.py`)
- 実データ（1,000台、約150万行）を用いた全車シミュレーションを実行（`outputs/criteria_step1_and_step2_evaluation.csv`）。
- **長期駐車（3日以上 / 対象850台）**:
  - 省電力 7h/日 以上達成: **850 / 850 台 (100.0%)**（平均 18.00 h/日、目標7hを大差でクリア）
  - 段階1クリア率: **760 / 850 台 (89.4%)**
- **段階2（突発利用除外時の全車達成）**:
  - 突発利用（過去365日に実績のない時間帯スロット）の発生率は **0.61%**。
  - 突発利用を除外した場合の解錠遅延: **全1,000台で 0 件（達成率 100.0%）** を完全実証！
  - 突発除外時の省電力理論値: **全車平均 20.51 h/日（暗電流削減効果: 31.61 mA）**。

---

## 5. okuya-x1pro-ubuntu 側での作業タスク（夜間および今後のステップ）

本日予定していたToDo（長期駐車検証、突発理論値算出、5大検討書ドラフト作成）が前倒しで全て完了しました。
今夜 `okuya-x1pro-ubuntu` で作業する際は、以下のステップへ進むことができます。

### (1) 環境同期と健全性確認
```bash
cd ~/develop/PSMP
git pull origin main
source .venv/bin/activate
pytest -v  # 全5件パス確認
```

### (2) 今後の拡張・ブラッシュアップタスク
1. **検討書ドラフトの最終確認・体裁調整**:
   - `docs/PROPOSAL_TECHNICAL_REPORTS_DRAFT.md` をプレビュー確認し、委託元への提出形式（Word/PowerPoint/Markdown）に合わせた最終調整。
2. **Streamlit UI アプリ (`app.py`) への反映（オプション）**:
   - 新設した「段階1・段階2の全車達成結果」「突発利用の除外トグル」を UI 上のタブでインタラクティブに可視化する画面の追加。
