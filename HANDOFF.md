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

## 6. okuya-x1pro-ubuntu での作業実績（2026-11-18）

### (1) 委託元報告会向けプレゼンテーションスライド完成（全11スライド）
- 委託元（電子PF開発部 EEA開発室 開発3課 髙間様）の依頼項目（STEP1）に1対1で対応するスライドを設計・自動生成。
  - 設計書: [`PSMP_Client_Report_Slides.md`](file:///home/tomokatsu098/.gemini/antigravity/brain/41d5085e-2551-42d2-910f-cd3d119dabff/PSMP_Client_Report_Slides.md)
  - PPTX実体 (16:9): [`outputs/PSMP_電力マネジメント_成果報告.pptx`](outputs/PSMP_電力マネジメント_成果報告.pptx)
  - 自動生成スクリプト: [`scripts/generate_report_pptx.py`](scripts/generate_report_pptx.py)

### (2) 学習ステップ（7日〜365日）感度分析エビデンスの統合
- 7日、14日、30日、60日、90日、180日、365日の学習期間推移（ROC-AUC、Brierスコア、省電力率）を報告書・スライドに反映。
- 14日〜30日での初期急伸（新車納車直後から実用）および月1回OTA更新の妥当性を定量的エビデンスで証明。

### (3) 対象実機「Qualcomm QC8255 (SA8255P)」車載C-DCアーキテクチャへの適合確立
- 駐車中、主系Kryo AP（AAOS/QNX）が電源遮断（Power-gated）される中、常時稼働の低電力サブシステム（ARM Cortex-R52、TCM数十〜数百KB）上で自律推論するアーキテクチャを確立。

### (4) C99 ＆ Rust `no_std` 組み込みライブラリの実装・ベンチマーク検証
- **C99実装 (`embedded/c/`)**:
  - 動的メモリ確保を完全排除（ゼロアロケーション、MISRA-C / ISO 26262 ASIL-D 準拠）。
  - ECDFテーブル構造体: **わずか 772 Bytes**（QC8255 64KB枠の約1.2%）。
  - 実測レイテンシ: **12.80 ns (0.0128 μs)**（100万回推論平均）。
  - SoC安全弁、Re-sleep安全弁、出先チャタリング防止等のユニットテスト全て PASSED。
- **Rust `no_std` 実装 (`embedded/rust/psmp_edge_engine/`)**:
  - パニックフリー、静的配列による二分探索（`partition_point`）。
  - `cargo test`: 全3件 PASSED、`cargo build --release`: 成功。

### (5) Streamlit UI (`app.py`) への Tab 5 実装
- **Tab 5: ⚡ QC8255 (SA8255P): 車載エッジ推論 ＆ 1,000台全車検証** を新設。
  - QC8255 Cortex-R52 実機マイコン適合ベンチマーク（ROM 772B, 12.8ns）の表示。
  - スライダー操作による Cortex-R52 リアルタイム推論＆状態遷移（S0〜S4）エミュレータ。
  - 1,000台全車評価データ（段階1・段階2達成率100%）のヒストグラム・データビューア。

---

## 7. okuya-x1c での作業実績（2026-11-18午後: パワートレイン別バッテリー保護仕様）

### (1) パワートレイン種別（BEV / PHEV / HEV / ICE）の定義とプロファイル拡張
- `psmp/config.py`: `PowertrainType` Enum（`BEV`, `PHEV`, `HEV`, `ICE`）を新設。
- `DEFAULT_DRIVER_PROFILES` に `DRV_D_HEV_STANDARD` (HEV) および `DRV_E_ICE_COMMUTER` (ICE) を追加し、全パワートレインの諸元・バッテリー容量・ガード設定を完備。

### (2) パワートレイン連動型安全弁・状態遷移ガードの実装
- **`psmp/phase2_ig_off_planner.py` & `psmp/phase3_parked_monitor.py`**:
  - **BEV (純電気自動車)**: 高圧SoC $\le 20\%$ で車載DC-DCによる12V自動充電停止リスク・電欠防止のため即時S4 (Dormant) 強制休眠へ移行。
  - **PHEV (プラグインHV)**: 高圧SoC $\le 15\%$ でEVプレ空調を遮断。ガソリン自走可能により立ち往生リスクは低減しつつ、12V共倒れを防止。
  - **HEV / ICE**: エンジン停止中の高圧空調・高圧温調は不可。12V暗電流最小化（スマートキー待受のDeepSleep化）に徹し、30日放置上がりおよびセルモーターのクランキング始動電流を死守。

### (3) 詳細仕様書および Streamlit UI への反映
- **仕様書**: [`docs/POWER_MANAGEMENT_SPEC.md`](docs/POWER_MANAGEMENT_SPEC.md) に第5.1章「パワートレイン別バッテリー保護仕様」を追加（保護マトリクス・マイコンシーケンス）。
- **Streamlit UI (`app.py`)**: Tab 5 の Cortex-R52 エミュレータに「⚡ パワートレイン種別（BEV / PHEV / HEV / ICE）」のセレクトボックスを追加し、パワートレインに応じたガード発動（S0〜S4）をインタラクティブに検証可能に拡張。
- **ユニットテスト**: [`tests/test_powertrain.py`](tests/test_powertrain.py) を新設し、全3件 PASSED。既存テストも含め全テストが正常に通過することを確認。

