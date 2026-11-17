# deep7.py

駐車中の車が「何時間後に乗られるか」を予測し、乗られる確率が低い時間帯だけを DeepSleep にする。
予算・ガードなし、全体で1つの閾値で判定する。検証期間で DeepSleep が目標（既定 7 h/台日）以上になる最小の閾値を選び、テスト期間で評価する。
1ファイルで完結する。入力 CSV は実データでも疑似データでもよく、下の「入力 CSV」の列があれば動く。

## 使い方

```
python deep7.py <CSVファイル>
python deep7.py <CSVファイル> --target 7 --sample 20 --no-cache
```

| オプション | 意味 |
|---|---|
| `--target` | 検証期間の DeepSleep の目標 [h/台日]。既定 7 |
| `--sample N` | 各 `data_group` の先頭 N 台だけで動作確認する（`data_group` 列がなければ全体の先頭 N 台） |
| `--no-cache` | キャッシュを使わず、特徴量・学習・推論をやり直す |

進捗は工程 1/6〜6/6 と、経過・残り時間つきのバーで出る。

## 入力 CSV

**必須の列**（なければエラーで止まる）

| 列 | 使い方 |
|---|---|
| `vehicle_id` | 車両の区別 |
| `event_date` | 期間の分割 |
| `ig_on_time`, `ig_off_time` | 駐車時間（次の `ig_on_time` − 今の `ig_off_time`）の計算 |

**あれば使う列**（なければ欠損として扱い、そのまま動く）

| 列 | 使い方 | 欠けているとき |
|---|---|---|
| `weather` | 天気コード（晴れ 0 / 曇り 1 / 雨 2 / 強雨 3） | −1（不明） |
| `location_at_ig_off` | 場所コード（自宅 0 / 会社 1 / その他 2） | −1（不明） |
| `dow`, `is_weekend`, `is_holiday`, `trip_seq`, `temp_c` | そのまま特徴量 | 0 |
| `end_lat`, `end_lon` | 駐車位置（前の移動の到着地）。0.05° に丸めて区画にする。位置の精度はデータ次第 | 位置の特徴量は欠損、`cell_hist_*` は全体の履歴で代替 |
| `data_group` | `--sample` のグループ分けだけに使う | 全体で1グループ |

使わない列: 上の表にないすべての列（`start_lat`, `start_lon`、移動距離、湿度、天気の詳細、SOC・充電接続など）。
`parking_h` と `next_ig_on_time` は正解の計算にだけ使い、特徴量には入れない（コード内で確認している）。

## 駐車エピソード

車両ごとに `ig_off_time` で並べ、1件の移動の終了から次の移動の開始までを1つの駐車とする。最後の移動は次がないので除く。`parking_h` が 0 以下のものも除く。

## 期間の分割

CSV の `event_date` の最大日から逆算した、全車両共通の境界。

| 期間 | 長さ | 用途 |
|---|---|---|
| 学習 | 12か月 | モデルの学習 |
| 検証 | 3か月 | 早期終了と、閾値の選択 |
| テスト | 3か月 | 最終評価 |

学習・検証では、次の乗車が次の期間にかかる駐車を除く。

## 特徴量（102列）

どの特徴量も、その駐車より前の情報だけから作る。

### 1. 駐車開始時点（45列）

| 分類 | 列 |
|---|---|
| 駐車開始の時刻 | `off_hour`, `off_hour_sin`, `off_hour_cos`, `off_dow`, `off_month` |
| カレンダー・状況 | `dow`, `is_weekend`, `is_holiday`, `trip_seq`, `temp_c`, `weather_code`, `loc_code` |
| 駐車時間の履歴 | `hist_mean_h`, `hist_median_h`, `hist_q25_h`, `hist_q75_h`, `hist_count` |
| 長時間駐車の割合 | `hist_long24_rate`, `hist_long48_rate`, `hist_long72_rate` |
| 直近10件 | `hist_recent_mean_h`, `hist_recent_long24_rate`, `hist_recent_long48_rate`, `trend_long24_shift`（直近と全体の差） |
| 直前の駐車 | `prev_parking_h`, `prev2_parking_h` |
| 曜日別・時間帯別 | `hist_same_dow_mean_h`（同じ曜日の平均）, `hist_same_hour_mean_h`（駐車開始が同じ時間帯 ±2h の平均） |
| 次に乗られた時刻 | `hist_next_use_hour_med`, `hist_next_use_hour_std`（全体）, `hist_same_dow_next_hour_med`, `hist_same_dow_next_hour_std`（同じ曜日）。時刻は円周平均 |
| 場所ラベル別 | `loc_hist_mean_h`, `loc_hist_long24_rate`, `loc_hist_count`（今回と同じラベルの過去の駐車） |
| 過去の同じ時刻 | `used_same_slot_1w`〜`4w`, `same_slot_hit_rate_4w`（1〜4週前の同じ曜日に、同じ時間帯 ±2h で駐車を始めていたか。乗られた時刻ではなく、駐車を始めた時刻） |
| 駐車位置 | `park_lat`, `park_lon`（0.05° 丸め）, `cell_hist_count`, `cell_hist_mean_h`, `cell_hist_long24_rate`（同じ車・同じ0.05°区画での過去の駐車） |

初回など過去がない場合の既定値: 駐車時間の平均・中央値 12h、25%点 6h、75%点 24h、時刻の標準偏差 6h、割合 0。
緯度経度が欠けている駐車は、区画の履歴を持たない扱い（`cell_hist_*` は全体の履歴で代替）。

### 2. ブロックの情報（9列）

予測するブロック（0〜24h / 24〜48h / 48〜72h / 72h以降）の開始時点の情報。

`elapsed_h`, `elapsed_days`（駐車開始からの経過）, `current_hour`, `current_hour_sin`, `current_hour_cos`, `current_dow`, `current_month`, `current_is_weekend`, `current_is_holiday`

### 3. 窓に合わせた個人の利用確率（48列）

- `up_rot_0`〜`up_rot_23`: 次に乗られる時刻の分布（その車の過去全体。`(回数 + 0.5) / (件数 + 12)`）を、ブロック開始時刻から j 時間後の窓の順に並べ替えたもの。
- `up_rotdt_0`〜`up_rotdt_23`: 同じ分布を平日・休日に分けて持ち、各窓の日が平日か休日かに応じて選んだもの。

## 予測するもの

- ブロックごとに LightGBM の 25 クラス分類モデルを1つ作る（`0_24`, `24_48`, `48_72`, `72_PLUS`）。
- クラス 0〜23 = ブロック開始から何時間後の窓で乗られるか。クラス 24 = このブロックの24時間では乗られない。
- 学習では、駐車を30日で打ち切る。72h 以降のブロックは1駐車あたり最大12本に間引く。
- 設定: 学習率 0.05、葉 31、`min_data_in_leaf` 100、`lambda_l2` 1、最大 1000 ラウンド、検証での早期終了 50 ラウンド。
- 推論は駐車の全ブロックについて、24窓の確率を出す。

## 判定と評価

- **判定**: 窓の確率が閾値より低ければ、その窓を DeepSleep にする。閾値は全ブロック・全時間帯で共通。
- **閾値の候補**: 0.02%, 0.03%, 0.05%, 0.07%, 0.1%, 0.15%, 0.2%, 0.3%, 0.5%, 0.7%, 1%, 1.5%, 2%, 3%, 5%, 7.5%, 10%, 15%, 20%, 30%, 50%（21通り）。
- **選び方**: 検証期間で DeepSleep が目標以上になる最小の閾値。届かなければ最大の閾値を使い、注意を表示する。
- **取りこぼし率**: 実際に乗られた窓が DeepSleep になっていた駐車の割合。
- **DeepSleep [h/台日]**: DeepSleep にした時間の合計 ÷ 車両の台日数。

## 出力

`result_deep7_<CSV名>/`（`--sample` のときは `_sN` つき）

| ファイル | 内容 |
|---|---|
| `summary.txt` | 期間、選んだ閾値、検証・テストの取りこぼし率と DeepSleep、閾値ごとの結果、駐車時間別の内訳 |
| `thresholds.csv` | 21通りの閾値ごとの、検証・テストの取りこぼし率と DeepSleep |
| `by_parking.csv` | テスト期間の駐車時間別（2h未満 / 2-4h / 4-8h / 8-24h / 24-48h / 48h以上）の件数と取りこぼし件数 |

## キャッシュ

`cache_deep7/`（バージョン `v1`）

| ファイル | 内容 |
|---|---|
| `<CSV名>_episodes_v1.pkl` | 駐車エピソードと特徴量、期間の境界 |
| `<CSV名>_model_<ブロック>_v1.txt` | 学習済みモデル |
| `<CSV名>_probs_valid_v1.npz`, `<CSV名>_probs_test_v1.npz` | 検証・テストの確率 |

特徴量の定義を変えたら `--no-cache` で作り直すか、`deep7.py` 内の `VERSION` を変える。

## 注意

- 位置を粗く丸めたデータ（0.05° ≒ 約5km）では、自宅・会社・その他が同じ区画に入る車両があり、`cell_hist_*` では場所を区別できない。
- 実データの結果には車両 ID や位置が含まれることがあるので、社外に出さないこと。
