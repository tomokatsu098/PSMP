"""DeepSleepの寝起き判定（予算・ガードなし、全セル共通の1つの閾値）。
検証期間でDeepSleepが目標[h/台日]以上になる最小の閾値を選び、テスト期間で評価する。
予測モデルの入力には、駐車位置（end_lat/end_lon・0.05度丸め）と、同じ車・同じ位置での過去の駐車履歴を含む。

  python deep7.py real1000_A_lowsleep_miss30_long30.csv
  python deep7.py real1000_A_lowsleep_miss30_long30.csv --target 7 --sample 20 --no-cache

  --target    検証期間のDeepSleepの目標[h/台日]（既定 7）
  --sample N  各組の先頭N台だけで動作確認する
  --no-cache  キャッシュを使わず、特徴量・学習・推論をやり直す
出力: result_deep7_<CSV名>/（summary.txt・thresholds.csv・by_parking.csv）。キャッシュ: cache_deep7/
"""
import argparse
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--target", type=float, default=7.0)
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--no-cache", action="store_true")
ARGS = ap.parse_args()

NAME = os.path.splitext(os.path.basename(ARGS.csv))[0] + (f"_s{ARGS.sample}" if ARGS.sample else "")
CACHE_DIR = "cache_deep7"
OUT_DIR = f"result_deep7_{NAME}"
VERSION = "v1"

N_WIN = 24
N_CLASS = N_WIN + 1
BLOCK_GROUPS = ("0_24", "24_48", "48_72", "72_PLUS")
N_GROUPS = len(BLOCK_GROUPS)
TRAIN_MONTHS, VALID_MONTHS, TEST_MONTHS = 12, 3, 3
MAX_PARKING_H = 24 * 30
MAX_LONG_BLOCKS_PER_EPISODE = 12
MIN_ROWS_FOR_GROUP_TRAIN = 20
NUM_BOOST_ROUND_MAX = 1000
EARLY_STOPPING_ROUNDS = 50
ROW_CHUNK = 200_000
NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 86_400_000_000_000
THRESHOLDS = np.array([0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007,
                       0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50], dtype=np.float32)
PARAMS = {
    "objective": "multiclass", "num_class": N_CLASS, "metric": "multi_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "max_bin": 127,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_data_in_leaf": 100, "lambda_l2": 1.0, "force_col_wise": True,
    "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 42,
    "feature_fraction_seed": 42, "bagging_seed": 42,
}

WEATHER_MAP = {"晴れ": 0, "曇り": 1, "雨": 2, "強雨": 3}
LOC_MAP = {"自宅": 0, "会社": 1, "その他": 2}
LOAD_COLUMNS = ("vehicle_id", "data_group", "event_date", "ig_on_time", "ig_off_time", "weather",
                "location_at_ig_off", "dow", "is_weekend", "is_holiday", "trip_seq", "temp_c", "end_lat", "end_lon")
UP_ALL_COLS = [f"hist_use_prob_{h}" for h in range(24)]
UP_WD_COLS = [f"use_prob_wd_{h}" for h in range(24)]
UP_WE_COLS = [f"use_prob_we_{h}" for h in range(24)]

SCALAR_FEATURES = [
    "off_hour", "off_hour_sin", "off_hour_cos", "off_dow", "off_month",
    "dow", "is_weekend", "is_holiday", "trip_seq", "temp_c", "weather_code", "loc_code",
    "hist_mean_h", "hist_median_h", "hist_q25_h", "hist_q75_h",
    "hist_long24_rate", "hist_long48_rate", "hist_long72_rate",
    "hist_same_dow_mean_h", "hist_same_hour_mean_h", "hist_count",
    "hist_recent_mean_h", "hist_recent_long24_rate", "hist_recent_long48_rate",
    "prev_parking_h", "prev2_parking_h", "trend_long24_shift",
    "hist_next_use_hour_med", "hist_next_use_hour_std",
    "hist_same_dow_next_hour_med", "hist_same_dow_next_hour_std",
    "used_same_slot_1w", "used_same_slot_2w", "used_same_slot_3w", "used_same_slot_4w", "same_slot_hit_rate_4w",
    "loc_hist_mean_h", "loc_hist_long24_rate", "loc_hist_count",
    "park_lat", "park_lon", "cell_hist_count", "cell_hist_mean_h", "cell_hist_long24_rate",
]
BLOCK_TIME_FEATURES = ["elapsed_h", "elapsed_days", "current_hour", "current_hour_sin", "current_hour_cos",
                       "current_dow", "current_month", "current_is_weekend", "current_is_holiday"]
FEATURES = (SCALAR_FEATURES + BLOCK_TIME_FEATURES
            + [f"up_rot_{j}" for j in range(24)] + [f"up_rotdt_{j}" for j in range(24)])
N_FEATURES, N_SCALAR = len(FEATURES), len(SCALAR_FEATURES)
assert not {"parking_h", "next_ig_on_time", "start_lat", "start_lon"} & set(FEATURES)


# ============================================================
# 進捗表示
# ============================================================
T_START = time.time()
STAGES = ("CSVの読み込み", "駐車エピソードと特徴量", "期間の分割", "モデルの学習", "確率の推論", "閾値の選択と評価")


def fmt(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def stage(i, note=""):
    print(f"\n■ 工程 {i}/{len(STAGES)}: {STAGES[i - 1]}{note}（開始から {fmt(time.time() - T_START)}）", flush=True)


class Bar:
    def __init__(self, label, total):
        self.label, self.total, self.t0, self.last = label, max(int(total), 1), time.time(), 0.0

    def update(self, done, extra=""):
        now = time.time()
        if done < self.total and now - self.last < 0.5:
            return
        self.last = now
        frac = min(done / self.total, 1.0)
        el = now - self.t0
        eta = el / frac - el if frac > 0 else 0
        sys.stdout.write(f"\r  {self.label} |{'#' * int(30 * frac):-<30}| {frac * 100:5.1f}%"
                         f"  経過 {fmt(el)}  残り {fmt(eta)}  {extra}   ")
        if done >= self.total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def close(self, extra=""):
        sys.stdout.write(f"\r  {self.label} |{'#' * 30}| 完了  所要 {fmt(time.time() - self.t0)}  {extra}"
                         + " " * 20 + "\n")
        sys.stdout.flush()


class Step:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"  ・{self.label}…", end="", flush=True)

    def __exit__(self, *exc):
        print(f" 完了（{fmt(time.time() - self.t0)}）", flush=True)


def chunks(total, size=ROW_CHUNK):
    for lo in range(0, total, size):
        yield lo, min(lo + size, total)


def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{NAME}_{name}_{VERSION}")


# ============================================================
# 祝日
# ============================================================
def _equinox_day(year, base):
    return int(base + 0.242194 * (year - 1980) - int((year - 1980) / 4))


def _nth_monday(year, month, n):
    d = date(year, month, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7 + 7 * (n - 1))


def japan_holidays(years):
    days = set()
    for y in years:
        days |= {date(y, 1, 1), _nth_monday(y, 1, 2), date(y, 2, 11), date(y, 2, 23),
                 date(y, 3, _equinox_day(y, 20.8431)), date(y, 4, 29), date(y, 5, 3), date(y, 5, 4),
                 date(y, 5, 5), _nth_monday(y, 7, 3), date(y, 8, 11), _nth_monday(y, 9, 3),
                 date(y, 9, _equinox_day(y, 23.2488)), _nth_monday(y, 10, 2), date(y, 11, 3), date(y, 11, 23)}
    for d in sorted(days):
        if d.weekday() == 6:
            nd = d + timedelta(days=1)
            while nd in days:
                nd += timedelta(days=1)
            days.add(nd)
    for d in sorted(days):
        nd, nnd = d + timedelta(days=1), d + timedelta(days=2)
        if nd not in days and nnd in days and nd.weekday() != 6:
            days.add(nd)
    return days


def build_calendar_lut(min_ord, max_ord):
    dates = pd.to_datetime(np.arange(min_ord, max_ord + 1), unit="D")
    holidays = japan_holidays(range(dates.year.min() - 1, dates.year.max() + 2))
    holiday = np.array([d.date() in holidays for d in dates], dtype=np.float32)
    return int(min_ord), dates.month.to_numpy().astype(np.float32), holiday


# ============================================================
# 読み込みと特徴量
# ============================================================
def read_csv(path):
    size = os.path.getsize(path)
    bar = Bar("CSV", size)
    parts = []
    with open(path, "rb") as fh:
        for part in pd.read_csv(fh, usecols=lambda c: c in LOAD_COLUMNS, dtype={"vehicle_id": str}, chunksize=500_000):
            parts.append(part)
            bar.update(fh.tell(), f"{sum(len(p) for p in parts):,}行")
    df = pd.concat(parts, ignore_index=True)
    missing = {"vehicle_id", "event_date", "ig_on_time", "ig_off_time"} - set(df.columns)
    if missing:
        raise SystemExit(f"必須の列がありません: {sorted(missing)}")
    for c in LOAD_COLUMNS:
        if c not in df and c != "data_group":
            df[c] = np.nan
    bar.close(f"{len(df):,}行 / 車両 {df['vehicle_id'].nunique():,}台")
    return df


def calendar_bounds(event_date):
    test_end = event_date.max().normalize() + pd.Timedelta(days=1)
    test_start = test_end - pd.DateOffset(months=TEST_MONTHS)
    valid_start = test_start - pd.DateOffset(months=VALID_MONTHS)
    train_start = valid_start - pd.DateOffset(months=TRAIN_MONTHS)
    return train_start, valid_start, test_start, test_end


def build_episodes(df):
    with Step("日時の変換と並べ替え"):
        for c in ("event_date", "ig_on_time", "ig_off_time"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.sort_values(["vehicle_id", "ig_off_time"]).reset_index(drop=True)
        df["next_ig_on_time"] = df.groupby("vehicle_id")["ig_on_time"].shift(-1)
        df["parking_h"] = (df["next_ig_on_time"] - df["ig_off_time"]).dt.total_seconds() / 3600.0
        ok = df["parking_h"].notna() & (df["parking_h"] > 0) & df["ig_off_time"].notna()
        df = df[ok].reset_index(drop=True)
    print(f"    駐車エピソード {len(df):,}件")
    off = df["ig_off_time"]
    df["off_hour"] = off.dt.hour + off.dt.minute / 60.0
    df["off_hour_sin"] = np.sin(2 * np.pi * df["off_hour"] / 24)
    df["off_hour_cos"] = np.cos(2 * np.pi * df["off_hour"] / 24)
    df["off_dow"] = off.dt.dayofweek
    df["off_month"] = off.dt.month
    df["weather_code"] = df["weather"].map(WEATHER_MAP).fillna(-1).astype(int)
    df["loc_code"] = df["location_at_ig_off"].map(LOC_MAP).fillna(-1).astype(int)
    for c in ("dow", "is_weekend", "is_holiday", "trip_seq", "temp_c"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return add_causal_history(df)


def _slot_hits(times, hour_bin, starts, ends):
    """1〜4週前の同じ曜日・同じ時刻（±2h）に駐車を始めていたか。"""
    hits = np.zeros((len(times), 4), dtype=np.float32)
    offsets = np.arange(-2, 3)
    bar = Bar("1〜4週前の同じ時刻", len(starts))
    for k, (s, e) in enumerate(zip(starts, ends), 1):
        m = e - s
        if m > 1:
            tv, hv = times[s:e], hour_bin[s:e]
            counts = np.zeros((m + 1, 24), dtype=np.int32)
            counts[np.arange(1, m + 1), hv] = 1
            np.cumsum(counts, axis=0, out=counts)
            pos = np.arange(m)
            bins5 = (hv[:, None] + offsets) % 24
            for wk in range(1, 5):
                target = tv - np.int64(wk * 7) * NS_PER_DAY
                lo = np.searchsorted(tv, target - NS_PER_DAY, side="left")
                hi = np.minimum(np.searchsorted(tv, target + NS_PER_DAY, side="right"), pos)
                lo = np.minimum(lo, hi)
                hits[s:e, wk - 1] = np.take_along_axis(counts[hi] - counts[lo], bins5, axis=1).sum(axis=1) > 0
        bar.update(k, f"{k:,}/{len(starts):,}台")
    bar.close()
    return hits


def add_causal_history(df):
    """各駐車の時点で、その車の過去の駐車だけから作る特徴量。"""
    n = len(df)
    veh = df["vehicle_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, veh[1:] != veh[:-1]])
    ends = np.r_[starts[1:], n]
    group_start = np.repeat(starts, ends - starts)
    idx = np.arange(n)
    cnt = (idx - group_start).astype(np.float64)
    has_past = cnt > 0
    safe_cnt = np.maximum(cnt, 1.0)
    parking = df["parking_h"].to_numpy(dtype=np.float64)
    hour_bin = np.mod(np.floor(df["off_hour"].to_numpy(dtype=np.float64)).astype(np.int64), 24)
    dow = df["off_dow"].to_numpy().astype(np.int64) % 7
    loc_bin = np.clip(df["loc_code"].to_numpy().astype(np.int64) + 1, 0, 3)
    off_ns = df["ig_off_time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    next_on = df["next_ig_on_time"]
    next_hour = (next_on.dt.hour + next_on.dt.minute / 60.0).to_numpy(dtype=np.float64)
    next_bin = np.mod(np.floor(next_hour).astype(np.int64), 24)
    next_is_weekend = next_on.dt.dayofweek.to_numpy() >= 5

    def prefix(values):
        gx = np.empty(n + 1, dtype=np.float64)
        gx[0] = 0.0
        np.cumsum(values, dtype=np.float64, out=gx[1:])
        return gx

    def past(values):
        gx = prefix(values)
        return gx[idx] - gx[group_start]

    out = {}
    with Step("駐車時間の履歴"):
        out["hist_mean_h"] = np.where(has_past, past(parking) / safe_cnt, 12.0)
        for h_thr, name in ((24, "hist_long24_rate"), (48, "hist_long48_rate"), (72, "hist_long72_rate")):
            out[name] = np.where(has_past, past((parking >= h_thr).astype(np.float64)) / safe_cnt, 0.0)
        out["hist_count"] = cnt
        lo10 = np.maximum(idx - 10, group_start)
        cnt10 = (idx - lo10).astype(np.float64)
        safe10 = np.maximum(cnt10, 1.0)
        for name, values, default in (("hist_recent_mean_h", parking, 12.0),
                                      ("hist_recent_long24_rate", (parking >= 24).astype(np.float64), 0.0),
                                      ("hist_recent_long48_rate", (parking >= 48).astype(np.float64), 0.0)):
            gx = prefix(values)
            out[name] = np.where(cnt10 > 0, (gx[idx] - gx[lo10]) / safe10, default)
        out["trend_long24_shift"] = out["hist_recent_long24_rate"] - out["hist_long24_rate"]
        out["prev_parking_h"] = np.where(has_past, parking[np.maximum(idx - 1, group_start)], 12.0)
        out["prev2_parking_h"] = np.where(cnt >= 2, parking[np.maximum(idx - 2, group_start)], out["prev_parking_h"])

    with Step("駐車時間の分位点"):
        shifted = np.full(n, np.nan)
        mask_shift = idx > group_start
        shifted[mask_shift] = parking[idx[mask_shift] - 1]
        exp = pd.Series(shifted).groupby(group_start).expanding(min_periods=1)

        def expanding(result):
            return result.reset_index(level=0, drop=True).sort_index().to_numpy()

        out["hist_median_h"] = np.nan_to_num(expanding(exp.median()), nan=12.0)
        out["hist_q25_h"] = np.nan_to_num(expanding(exp.quantile(0.25)), nan=6.0)
        out["hist_q75_h"] = np.nan_to_num(expanding(exp.quantile(0.75)), nan=24.0)

    with Step("次に乗られた時刻・曜日別・場所別の履歴"):
        angle = 2 * np.pi * next_hour / 24.0
        sin_v, cos_v = np.sin(angle), np.cos(angle)
        mean_sin, mean_cos = past(sin_v) / safe_cnt, past(cos_v) / safe_cnt
        resultant = np.clip(np.sqrt(mean_sin ** 2 + mean_cos ** 2), 1e-9, 1.0)
        out["hist_next_use_hour_med"] = np.where(
            has_past, (np.arctan2(mean_sin, mean_cos) % (2 * np.pi)) / (2 * np.pi) * 24.0, 12.0)
        out["hist_next_use_hour_std"] = np.where(has_past, np.sqrt(-2.0 * np.log(resultant)) / (2 * np.pi) * 24.0, 6.0)

        dow_cnt, dow_sum, dow_sin, dow_cos = (np.zeros(n) for _ in range(4))
        for d in range(7):
            sel = dow == d
            mk = sel.astype(np.float64)
            dow_cnt[sel] = past(mk)[sel]
            dow_sum[sel] = past(mk * parking)[sel]
            dow_sin[sel] = past(mk * sin_v)[sel]
            dow_cos[sel] = past(mk * cos_v)[sel]
        has_dow = dow_cnt > 0
        safe_dow = np.maximum(dow_cnt, 1.0)
        out["hist_same_dow_mean_h"] = np.where(has_dow, dow_sum / safe_dow, out["hist_mean_h"])
        ms, mc = dow_sin / safe_dow, dow_cos / safe_dow
        dow_r = np.clip(np.sqrt(ms ** 2 + mc ** 2), 1e-9, 1.0)
        out["hist_same_dow_next_hour_med"] = np.where(
            has_dow, (np.arctan2(ms, mc) % (2 * np.pi)) / (2 * np.pi) * 24.0, out["hist_next_use_hour_med"])
        out["hist_same_dow_next_hour_std"] = np.where(
            has_dow, np.sqrt(-2.0 * np.log(dow_r)) / (2 * np.pi) * 24.0, out["hist_next_use_hour_std"])

        sh_cnt, sh_sum = np.zeros(n), np.zeros(n)
        for b in range(24):
            mk = (hour_bin == b).astype(np.float64)
            near = np.abs(((hour_bin - b + 12) % 24) - 12) <= 2
            sh_cnt[near] += past(mk)[near]
            sh_sum[near] += past(mk * parking)[near]
        out["hist_same_hour_mean_h"] = np.where(sh_cnt > 0, sh_sum / np.maximum(sh_cnt, 1.0), out["hist_mean_h"])

        loc_cnt, loc_sum, loc_l24 = np.zeros(n), np.zeros(n), np.zeros(n)
        for b in range(4):
            sel = loc_bin == b
            mk = sel.astype(np.float64)
            loc_cnt[sel] = past(mk)[sel]
            loc_sum[sel] = past(mk * parking)[sel]
            loc_l24[sel] = past(mk * (parking >= 24))[sel]
        has_loc = loc_cnt > 0
        out["loc_hist_mean_h"] = np.where(has_loc, loc_sum / np.maximum(loc_cnt, 1.0), out["hist_mean_h"])
        out["loc_hist_long24_rate"] = np.where(has_loc, loc_l24 / np.maximum(loc_cnt, 1.0), 0.0)
        out["loc_hist_count"] = loc_cnt

    with Step("駐車位置と、同じ位置での過去の駐車"):
        lat = pd.to_numeric(df["end_lat"], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(df["end_lon"], errors="coerce").to_numpy(dtype=np.float64)
        has_cell = np.isfinite(lat) & np.isfinite(lon)
        cell = (np.rint(np.nan_to_num(lat) / 0.05).astype(np.int64) * 10_000
                + np.rint(np.nan_to_num(lon) / 0.05).astype(np.int64))
        key = np.where(has_cell, pd.factorize(veh)[0].astype(np.int64) * 100_000_000 + cell, -1 - idx)
        groups = pd.Series(parking).groupby(key)
        c_cnt = groups.cumcount().to_numpy(dtype=np.float64)
        c_sum = groups.cumsum().to_numpy() - parking
        long24 = (parking >= 24).astype(np.float64)
        c_l24 = pd.Series(long24).groupby(key).cumsum().to_numpy() - long24
        has_cell_past = c_cnt > 0
        out["park_lat"] = np.where(has_cell, lat, np.nan)
        out["park_lon"] = np.where(has_cell, lon, np.nan)
        out["cell_hist_count"] = c_cnt
        out["cell_hist_mean_h"] = np.where(has_cell_past, c_sum / np.maximum(c_cnt, 1.0), out["hist_mean_h"])
        out["cell_hist_long24_rate"] = np.where(has_cell_past, c_l24 / np.maximum(c_cnt, 1.0), 0.0)

    hits = _slot_hits(off_ns, hour_bin, starts, ends)
    for w in range(4):
        out[f"used_same_slot_{w + 1}w"] = hits[:, w]
    out["same_slot_hit_rate_4w"] = hits.mean(axis=1)

    with Step("個人利用確率（次に乗られる時刻の分布）"):
        wd, we = (~next_is_weekend).astype(np.float64), next_is_weekend.astype(np.float64)
        cnt_wd, cnt_we = past(wd), past(we)
        for b in range(24):
            mk = (next_bin == b).astype(np.float64)
            out[UP_ALL_COLS[b]] = (past(mk) + 0.5) / (cnt + 12.0)
            out[UP_WD_COLS[b]] = (past(mk * wd) + 0.5) / (cnt_wd + 12.0)
            out[UP_WE_COLS[b]] = (past(mk * we) + 0.5) / (cnt_we + 12.0)

    added = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, index=df.index)
    return pd.concat([df, added], axis=1)


def split_by_calendar(df, bounds):
    train_start, valid_start, test_start, test_end = bounds
    d, nxt = df["event_date"], df["next_ig_on_time"]
    train = df[(d >= train_start) & (d < valid_start) & (nxt < valid_start)].reset_index(drop=True)
    valid = df[(d >= valid_start) & (d < test_start) & (nxt < test_start)].reset_index(drop=True)
    test = df[(d >= test_start) & (d < test_end)].reset_index(drop=True)
    return train, valid, test


# ============================================================
# ブロック行の特徴量行列
# ============================================================
class EpisodeContext:
    def __init__(self, data):
        self.n = len(data)
        self.scalar = np.ascontiguousarray(data[SCALAR_FEATURES].to_numpy(dtype=np.float32))
        self.off_hour = data["off_hour"].to_numpy(dtype=np.float64)
        self.off_ord = data["ig_off_time"].to_numpy().astype("datetime64[ns]").astype(np.int64) // NS_PER_DAY
        self.parking = data["parking_h"].to_numpy(dtype=np.float64)
        self.up_all = np.ascontiguousarray(data[UP_ALL_COLS].to_numpy(dtype=np.float32))
        self.up_wd = np.ascontiguousarray(data[UP_WD_COLS].to_numpy(dtype=np.float32))
        self.up_we = np.ascontiguousarray(data[UP_WE_COLS].to_numpy(dtype=np.float32))
        margin = int(np.ceil(self.parking.max() / 24.0)) + 3
        self.cal_base, self.cal_month, self.cal_holiday = build_calendar_lut(
            int(self.off_ord.min()) - 1, int(self.off_ord.max()) + margin)


def block_group_code(block_start_h):
    return np.minimum(np.asarray(block_start_h) // 24, N_GROUPS - 1).astype(np.int64)


def _expand_blocks(n_blocks):
    ep = np.repeat(np.arange(len(n_blocks), dtype=np.int64), n_blocks)
    offset = np.arange(int(n_blocks.sum()), dtype=np.int64) - np.repeat(np.cumsum(n_blocks) - n_blocks, n_blocks)
    return ep, offset


def make_training_pairs(parking):
    """学習用。30日で打ち切り、72h以降のブロックは最大12本へ等間隔に間引く。"""
    n_blocks = np.floor(np.minimum(parking, MAX_PARKING_H) / 24.0).astype(np.int64) + 1
    ep, offset = _expand_blocks(n_blocks)
    n_long = np.maximum(n_blocks - 3, 0)[ep]
    rank = offset - 3
    k = MAX_LONG_BLOCKS_PER_EPISODE
    denom = np.maximum(n_long - 1, 1)
    picked = np.rint(np.rint(rank * (k - 1) / denom) * denom / (k - 1))
    keep = (rank < 0) | (n_long <= k) | (rank == picked)
    return ep[keep], offset[keep] * 24


def make_prediction_pairs(parking):
    ep, offset = _expand_blocks(np.floor(parking / 24.0).astype(np.int64) + 1)
    return ep, offset * 24


def build_block_matrix(ctx, ep_idx, block_start):
    rows = len(ep_idx)
    out = np.empty((rows, N_FEATURES), dtype=np.float32)
    window = np.arange(N_WIN, dtype=np.float64)
    for lo, hi in chunks(rows):
        ei, bs_i = ep_idx[lo:hi], block_start[lo:hi]
        bs = bs_i.astype(np.float64)
        out[lo:hi, :N_SCALAR] = ctx.scalar[ei]
        off_hour = ctx.off_hour[ei]
        total = off_hour + bs
        day_offset = np.floor(total / 24.0)
        current_hour = total - day_offset * 24.0
        current_ord = ctx.off_ord[ei] + day_offset.astype(np.int64)
        current_dow = (current_ord + 3) % 7
        lut = current_ord - ctx.cal_base
        c = N_SCALAR
        for values in (bs, bs / 24.0, current_hour, np.sin(2 * np.pi * current_hour / 24.0),
                       np.cos(2 * np.pi * current_hour / 24.0), current_dow, ctx.cal_month[lut],
                       current_dow >= 5, ctx.cal_holiday[lut]):
            out[lo:hi, c] = values
            c += 1
        cols = ((np.floor(off_hour).astype(np.int64) + bs_i) % 24)[:, None] + np.arange(24)
        cols %= 24
        out[lo:hi, c:c + 24] = np.take_along_axis(ctx.up_all[ei], cols, axis=1)
        win_dow = (ctx.off_ord[ei][:, None]
                   + np.floor((off_hour[:, None] + bs[:, None] + window) / 24.0).astype(np.int64) + 3) % 7
        out[lo:hi, c + 24:c + 48] = np.where(win_dow >= 5, np.take_along_axis(ctx.up_we[ei], cols, axis=1),
                                             np.take_along_axis(ctx.up_wd[ei], cols, axis=1))
    return out


def make_labels(ctx, ep_idx, block_start):
    wake = np.floor(ctx.parking[ep_idx] - block_start).astype(np.int64)
    return np.where((wake >= 0) & (wake < N_WIN), wake, N_WIN).astype(np.int32)


# ============================================================
# 学習と推論
# ============================================================
def train_models(train_ctx, valid_ctx):
    import lightgbm as lgb
    tr_ep, tr_bs = make_training_pairs(train_ctx.parking)
    va_ep, va_bs = make_training_pairs(valid_ctx.parking)
    tr_gc, va_gc = block_group_code(tr_bs), block_group_code(va_bs)
    models = {}
    for gi, group in enumerate(BLOCK_GROUPS):
        st, sv = tr_gc == gi, va_gc == gi
        print(f"  [{gi + 1}/{N_GROUPS}] ブロック {group}: 学習 {int(st.sum()):,}行 / 検証 {int(sv.sum()):,}行", flush=True)
        if st.sum() < MIN_ROWS_FOR_GROUP_TRAIN:
            print("    学習行が少ないためスキップ")
            models[group] = None
            continue
        with Step("特徴量行列を作成"):
            dtrain = lgb.Dataset(build_block_matrix(train_ctx, tr_ep[st], tr_bs[st]),
                                 label=make_labels(train_ctx, tr_ep[st], tr_bs[st]),
                                 feature_name=FEATURES, params=PARAMS, free_raw_data=True)
            callbacks = []
            valid_sets = None
            if sv.sum() >= MIN_ROWS_FOR_GROUP_TRAIN:
                valid_sets = [lgb.Dataset(build_block_matrix(valid_ctx, va_ep[sv], va_bs[sv]),
                                          label=make_labels(valid_ctx, va_ep[sv], va_bs[sv]),
                                          reference=dtrain, params=PARAMS)]
                callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
        bar = Bar("学習（最大1000ラウンド・早期終了あり）", NUM_BOOST_ROUND_MAX)
        best = {"loss": np.inf, "it": 0}

        def progress(env, bar=bar, best=best):
            it = env.iteration + 1
            if env.evaluation_result_list:
                loss = env.evaluation_result_list[0][2]
                if loss < best["loss"]:
                    best.update(loss=loss, it=it)
                bar.update(it, f"ラウンド {it}  検証logloss {loss:.4f}（最良 {best['it']}）")
            else:
                bar.update(it, f"ラウンド {it}")

        model = lgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND_MAX, valid_sets=valid_sets,
                          callbacks=callbacks + [progress])
        bar.close(f"採用ラウンド {model.best_iteration or model.current_iteration()}")
        model.save_model(cache_path(f"model_{group}") + ".txt", num_iteration=model.best_iteration)
        models[group] = model
    return models


def load_models():
    import lightgbm as lgb
    paths = {g: cache_path(f"model_{g}") + ".txt" for g in BLOCK_GROUPS}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {g: lgb.Booster(model_file=p) for g, p in paths.items()}


def predict(models, ctx, label):
    ep, bs = make_prediction_pairs(ctx.parking)
    gc = block_group_code(bs)
    probs = np.zeros((len(ep), N_WIN), dtype=np.float32)
    bar = Bar(label, len(ep))
    done = 0
    for gi, group in enumerate(BLOCK_GROUPS):
        sel = np.flatnonzero(gc == gi)
        if models.get(group) is None:
            done += len(sel)
            continue
        for lo, hi in chunks(len(sel), 100_000):
            rows = sel[lo:hi]
            raw = models[group].predict(build_block_matrix(ctx, ep[rows], bs[rows]))
            probs[rows] = np.asarray(raw, dtype=np.float32)[:, :N_WIN]
            done += len(rows)
            bar.update(done, f"{done:,}/{len(ep):,}行")
    bar.close(f"{len(ep):,}行")
    return np.column_stack([ep, bs]).astype(np.int64), probs


# ============================================================
# 評価（21通りの閾値を一度に計算）
# ============================================================
class Plan:
    """閾値 THRESHOLDS[k] で寝かせる窓 ⇔ 確率 < THRESHOLDS[k] ⇔ k >= kmin。"""

    def __init__(self, ep_df, meta, probs):
        assert meta[:, 0].max() < len(ep_df), "確率のキャッシュとエピソードが合わない。--no-cache で作り直すこと"
        order = np.lexsort((meta[:, 1], meta[:, 0]))
        self.ep, bs, probs = meta[order, 0], meta[order, 1], probs[order]
        self.parking = ep_df["parking_h"].to_numpy(dtype=np.float64)
        self.n_ep = len(ep_df)
        ws = bs[:, None] + np.arange(N_WIN)
        park_row = self.parking[self.ep][:, None]
        self.valid = ws < park_row
        self.dur = np.clip(np.minimum(ws + 1.0, park_row) - ws, 0.0, 1.0)
        self.kmin = np.searchsorted(THRESHOLDS, probs.ravel(), side="right").reshape(probs.shape)
        last = np.flatnonzero(np.r_[self.ep[1:] != self.ep[:-1], True])
        wake_row = np.zeros(self.n_ep, dtype=np.int64)
        wake_row[self.ep[last]] = last
        self.wake_k = self.kmin[wake_row, np.floor(self.parking).astype(np.int64) - bs[wake_row]]
        veh = pd.factorize(ep_df["vehicle_id"])[0]
        off_d = ep_df["ig_off_time"].to_numpy().astype("datetime64[ns]").astype(np.int64) // NS_PER_DAY
        next_d = ep_df["next_ig_on_time"].to_numpy().astype("datetime64[ns]").astype(np.int64) // NS_PER_DAY
        self.n_days = int((pd.Series(next_d).groupby(veh).max() - pd.Series(off_d).groupby(veh).min() + 1).sum())
        self.n_veh = int(veh.max()) + 1

    def curve(self):
        n_thr = len(THRESHOLDS)
        sleep = np.bincount(self.kmin[self.valid], weights=self.dur[self.valid], minlength=n_thr + 1).cumsum()[:n_thr]
        miss = np.bincount(self.wake_k, minlength=n_thr + 1).cumsum()[:n_thr]
        return miss / self.n_ep, sleep / self.n_days

    def at(self, k):
        sleep_row = np.where(self.valid & (self.kmin <= k), self.dur, 0.0).sum(axis=1)
        return self.wake_k <= k, np.bincount(self.ep, weights=sleep_row, minlength=self.n_ep)


# ============================================================
# 本体
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8")

    def say(text=""):
        print(text, flush=True)
        log.write(text + "\n")

    use_cache = not ARGS.no_cache
    ep_cache = cache_path("episodes") + ".pkl"
    print(f"入力: {ARGS.csv}  /  目標DeepSleep（検証）: {ARGS.target:g}h/台日  /  特徴量 {N_FEATURES}次元")
    if use_cache and os.path.exists(ep_cache):
        stage(1, "（キャッシュを使用）")
        stage(2, "（キャッシュを使用）")
        with Step("駐車エピソードを読み込み"):
            episodes, bounds = pd.read_pickle(ep_cache)
    else:
        stage(1)
        raw = read_csv(ARGS.csv)
        bounds = calendar_bounds(pd.to_datetime(raw["event_date"], errors="coerce"))
        if ARGS.sample:
            group = raw["data_group"] if "data_group" in raw else pd.Series("all", index=raw.index)
            keep = [v for _, g in raw.groupby(group) for v in sorted(g["vehicle_id"].unique())[:ARGS.sample]]
            raw = raw[raw["vehicle_id"].isin(keep)].reset_index(drop=True)
            print(f"    動作確認用に {len(keep)}台に限定")
        stage(2)
        episodes = build_episodes(raw)
        del raw
        with Step("キャッシュに保存"):
            pd.to_pickle((episodes, bounds), ep_cache)

    stage(3)
    train, valid, test = split_by_calendar(episodes, bounds)
    del episodes
    train_start, valid_start, test_start, test_end = bounds
    period = (f"学習 {train_start.date()}〜{(valid_start - pd.Timedelta(days=1)).date()} / "
              f"検証 {valid_start.date()}〜{(test_start - pd.Timedelta(days=1)).date()} / "
              f"テスト {test_start.date()}〜{(test_end - pd.Timedelta(days=1)).date()}")
    print(f"  {period}\n  駐車: 学習 {len(train):,}件 / 検証 {len(valid):,}件 / テスト {len(test):,}件")

    prob_cache = {tag: cache_path(f"probs_{tag}") + ".npz" for tag in ("valid", "test")}
    if use_cache and all(os.path.exists(p) for p in prob_cache.values()):
        stage(4, "（キャッシュを使用）")
        stage(5, "（キャッシュを使用）")
        loaded = {tag: np.load(p) for tag, p in prob_cache.items()}
        results = {tag: (loaded[tag]["meta"], loaded[tag]["probs"]) for tag in loaded}
    else:
        valid_ctx = EpisodeContext(valid)
        models = load_models() if use_cache else None
        if models is None:
            stage(4)
            models = train_models(EpisodeContext(train), valid_ctx)
        else:
            stage(4, "（学習済みモデルを使用）")
        del train
        stage(5)
        results = {"valid": predict(models, valid_ctx, "検証"), "test": predict(models, EpisodeContext(test), "テスト")}
        for tag, (meta, probs) in results.items():
            np.savez(prob_cache[tag], meta=meta, probs=probs)

    stage(6)
    with Step("取りこぼしとDeepSleepを計算"):
        plans = {"valid": Plan(valid, *results["valid"]), "test": Plan(test, *results["test"])}
        curves = {tag: p.curve() for tag, p in plans.items()}
    ok = np.flatnonzero(curves["valid"][1] >= ARGS.target)
    k = int(ok[0]) if len(ok) else len(THRESHOLDS) - 1

    say(f"入力: {ARGS.csv}（車両 {plans['test'].n_veh:,}台）")
    say(f"期間: {period}")
    say(f"判定: 予算なし・ガードなし。確率が閾値より低い窓をDeepSleepにする（全セル共通の閾値）")
    say(f"特徴量: {N_FEATURES}次元（駐車位置 park_lat/park_lon と、同じ車・同じ位置での過去の駐車 cell_hist_* を含む）")
    if not len(ok):
        say(f"[注意] 検証でDeepSleepが {ARGS.target:g}h/台日 に届く閾値がないため、最大の閾値を使う")
    say(f"\n選んだ閾値: {THRESHOLDS[k] * 100:.2f}%（検証でDeepSleepが {ARGS.target:g}h/台日 以上になる最小の閾値）")
    say(f"  {'期間':<6}{'駐車件数':>10}{'取りこぼし件数':>14}{'取りこぼし率':>12}{'DeepSleep[h/台日]':>20}")
    for tag, label in (("valid", "検証"), ("test", "テスト")):
        p, (miss, sleep) = plans[tag], curves[tag]
        say(f"  {label:<6}{p.n_ep:>10,}{int(round(miss[k] * p.n_ep)):>14,}{miss[k] * 100:>11.2f}%{sleep[k]:>19.2f}")

    rows = []
    say("\n閾値ごとの結果")
    say(f"  {'閾値':>8}{'検証 取りこぼし率':>16}{'DeepSleep':>11}{'テスト 取りこぼし率':>18}{'DeepSleep':>11}")
    for j, t in enumerate(THRESHOLDS):
        vm, vs, tm, ts = curves["valid"][0][j], curves["valid"][1][j], curves["test"][0][j], curves["test"][1][j]
        rows.append({"閾値": float(t), "検証_取りこぼし率": vm, "検証_DeepSleep": vs, "テスト_取りこぼし率": tm,
                     "テスト_DeepSleep": ts, "選択": j == k})
        say(f"  {t * 100:>7.2f}%{vm * 100:>15.2f}%{vs:>10.2f}h{tm * 100:>17.2f}%{ts:>10.2f}h" + ("  ← 選択" if j == k else ""))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "thresholds.csv"), index=False, encoding="utf-8-sig")

    labels = ["2h未満", "2-4h", "4-8h", "8-24h", "24-48h", "48h以上"]
    edges = [0, 2, 4, 8, 24, 48, np.inf]
    missed, _ = plans["test"].at(k)
    park_bin = pd.cut(plans["test"].parking, edges, labels=labels, right=False)
    n_miss = int(missed.sum())
    t = pd.DataFrame({"区分": park_bin, "missed": missed}).groupby("区分", observed=False)["missed"].agg(["size", "sum"])
    say(f"\nテスト期間の駐車時間別（取りこぼし {n_miss:,}件の内訳）")
    say(f"  {'駐車時間':<8}{'駐車件数':>10}{'取りこぼし件数':>14}{'割合':>8}{'取りこぼし率':>12}")
    for name, r in t.iterrows():
        say(f"  {name:<8}{int(r['size']):>10,}{int(r['sum']):>14,}{r['sum'] / max(n_miss, 1) * 100:>7.1f}%"
            f"{r['sum'] / max(r['size'], 1) * 100:>11.2f}%")
    t.rename(columns={"size": "駐車件数", "sum": "取りこぼし件数"}).to_csv(
        os.path.join(OUT_DIR, "by_parking.csv"), encoding="utf-8-sig")
    say(f"\n保存: {OUT_DIR}  （全体の所要 {fmt(time.time() - T_START)}）")
    log.close()


if __name__ == "__main__":
    main()
