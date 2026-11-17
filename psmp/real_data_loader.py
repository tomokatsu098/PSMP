"""
Real-world Trip Profile to PSMP ParkingEventRecord Adapter (`psmp/real_data_loader.py`).
Extracts parking sessions (IG-OFF -> IG-ON) from synthetic/real trip CSV files (syn548_profile_v4).
"""

from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from psmp.config import DriverPreferenceProfile, POIClusterSpec, POIType
from psmp.phase1_collector_learner import ParkingEventRecord


def calculate_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    2点間の直線距離 (km) を算出する (Haversine formula)
    """
    R = 6371.0  # 地球半径 km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def get_season_name(dt: datetime) -> str:
    month = dt.month
    if month in (6, 7, 8):
        return "summer"
    elif month in (12, 1, 2):
        return "winter"
    return "spring_autumn"


class RealTripDataLoader:
    """
    走行データ CSV (syn548_profile_v4) から PSMP 評価用の駐車セッションを生成するローダークラス
    """

    def __init__(self, train_csv_path: str, eval_csv_path: Optional[str] = None):
        self.train_csv_path = Path(train_csv_path)
        self.eval_csv_path = Path(eval_csv_path) if eval_csv_path else None
        self._train_df: Optional[pd.DataFrame] = None
        self._eval_df: Optional[pd.DataFrame] = None
        self.home_coords: Dict[str, Tuple[float, float]] = {}
        self.work_coords: Dict[str, Tuple[float, float]] = {}

    def _load_dataframes(self) -> None:
        if self._train_df is None:
            self._train_df = pd.read_csv(self.train_csv_path)
            self._train_df["ig_on_dt"] = pd.to_datetime(self._train_df["ig_on_time"])
            self._train_df["ig_off_dt"] = pd.to_datetime(self._train_df["ig_off_time"])
            self._train_df["event_dt"] = pd.to_datetime(self._train_df["event_date"])

        if self.eval_csv_path and self._eval_df is None:
            self._eval_df = pd.read_csv(self.eval_csv_path)
            self._eval_df["ig_on_dt"] = pd.to_datetime(self._eval_df["ig_on_time"])
            self._eval_df["ig_off_dt"] = pd.to_datetime(self._eval_df["ig_off_time"])
            self._eval_df["event_dt"] = pd.to_datetime(self._eval_df["event_date"])

        # 自宅・勤務先の座標キャッシュ
        for vid, group in self._train_df.groupby("vehicle_id"):
            homes = group[group["location_at_ig_off"] == "自宅"]
            if not homes.empty:
                self.home_coords[vid] = (float(homes["end_lat"].iloc[0]), float(homes["end_lon"].iloc[0]))
            works = group[group["location_at_ig_off"] == "会社"]
            if not works.empty:
                self.work_coords[vid] = (float(works["end_lat"].iloc[0]), float(works["end_lon"].iloc[0]))

    def extract_parking_records(
        self,
        df: pd.DataFrame,
        vehicle_ids: Optional[List[str]] = None,
        max_train_days: Optional[int] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        out_of_home_only: bool = True,
    ) -> Tuple[List[ParkingEventRecord], Dict[str, POIClusterSpec]]:
        """
        指定された条件で駐車セッションを抽出する
        """
        self._load_dataframes()

        target_df = df
        if vehicle_ids is not None:
            target_df = target_df[target_df["vehicle_id"].isin(vehicle_ids)]

        if min_date is not None:
            target_df = target_df[target_df["event_dt"] >= pd.to_datetime(min_date)]
        if max_date is not None:
            target_df = target_df[target_df["event_dt"] <= pd.to_datetime(max_date)]

        if max_train_days is not None:
            start_date = target_df["event_dt"].min()
            cutoff = start_date + pd.Timedelta(days=max_train_days)
            target_df = target_df[target_df["event_dt"] < cutoff]

        records: List[ParkingEventRecord] = []
        poi_clusters: Dict[str, POIClusterSpec] = {}

        # 車両ごとにソートしてセッション化
        for vid, v_df in target_df.groupby("vehicle_id"):
            v_sorted = v_df.sort_values("ig_on_dt").copy()
            v_sorted["prev_off_dt"] = v_sorted["ig_off_dt"].shift(1)
            v_sorted["prev_loc"] = v_sorted["location_at_ig_off"].shift(1)
            v_sorted["prev_end_lat"] = v_sorted["end_lat"].shift(1)
            v_sorted["prev_end_lon"] = v_sorted["end_lon"].shift(1)
            v_sorted["prev_temp"] = v_sorted["temp_c"].shift(1)

            home_lat, home_lon = self.home_coords.get(vid, (35.0, 137.0))

            for idx, row in v_sorted.iterrows():
                if pd.isna(row["prev_off_dt"]):
                    continue

                prev_loc = row["prev_loc"]
                if out_of_home_only and prev_loc == "自宅":
                    continue

                park_start: datetime = row["prev_off_dt"].to_pydatetime()
                departure: datetime = row["ig_on_dt"].to_pydatetime()
                duration_min = (departure - park_start).total_seconds() / 60.0

                # 異常に短すぎる駐車（1分未満）や1ヶ月を超える超長期は除外またはクランプ
                if duration_min < 2.0 or duration_min > 43200.0:  # 30日超
                    continue

                p_lat = float(row["prev_end_lat"])
                p_lon = float(row["prev_end_lon"])

                # POI-ID決定
                if prev_loc == "会社":
                    poi_id = f"POI_{vid}_WORK"
                    poi_type = POIType.OFFICE_WORK
                    sec_risk = 0.20
                    base_m = 480.0
                elif prev_loc == "自宅":
                    poi_id = f"POI_{vid}_HOME"
                    poi_type = POIType.HOME
                    sec_risk = 0.05
                    base_m = 720.0
                else:
                    poi_id = f"POI_{vid}_{p_lat:.2f}_{p_lon:.2f}"
                    poi_type = POIType.SUPERMARKET
                    sec_risk = 0.45
                    base_m = 45.0

                dist_home = calculate_haversine_distance_km(p_lat, p_lon, home_lat, home_lon)

                if poi_id not in poi_clusters:
                    poi_clusters[poi_id] = POIClusterSpec(
                        poi_id=poi_id,
                        name_ja=f"{vid}訪問先 ({prev_loc})",
                        poi_type=poi_type,
                        lat=p_lat,
                        lon=p_lon,
                        distance_to_home_km=dist_home,
                        security_risk_level=sec_risk,
                        base_mean_minutes=base_m,
                        base_std_minutes=max(10.0, base_m * 0.25),
                    )

                temp_val = float(row["prev_temp"]) if not pd.isna(row["prev_temp"]) else 20.0

                record = ParkingEventRecord(
                    session_id=f"SES_{vid}_{len(records):05d}",
                    driver_id=vid,
                    poi_id=poi_id,
                    lat=p_lat,
                    lon=p_lon,
                    park_start_dt=park_start,
                    departure_dt=departure,
                    duration_minutes=round(duration_min, 1),
                    day_of_week=park_start.weekday(),
                    is_holiday=bool(row["is_holiday"] == 1 or row["is_weekend"] == 1),
                    hour_of_day=park_start.hour,
                    season=get_season_name(park_start),
                    ambient_temp_c=round(temp_val, 1),
                    has_passengers=False,
                    initial_soc_pct=72.0,
                    distance_to_home_km=dist_home,
                    anomaly_type="normal",
                    is_home=(prev_loc == "自宅"),
                )
                records.append(record)

        return records, poi_clusters

    def get_train_records(
        self,
        vehicle_ids: Optional[List[str]] = None,
        max_train_days: Optional[int] = None,
        out_of_home_only: bool = True,
    ) -> Tuple[List[ParkingEventRecord], Dict[str, POIClusterSpec]]:
        self._load_dataframes()
        assert self._train_df is not None
        return self.extract_parking_records(
            self._train_df,
            vehicle_ids=vehicle_ids,
            max_train_days=max_train_days,
            out_of_home_only=out_of_home_only,
        )

    def get_eval_records(
        self,
        vehicle_ids: Optional[List[str]] = None,
        out_of_home_only: bool = True,
    ) -> Tuple[List[ParkingEventRecord], Dict[str, POIClusterSpec]]:
        self._load_dataframes()
        assert self._eval_df is not None
        return self.extract_parking_records(
            self._eval_df,
            vehicle_ids=vehicle_ids,
            out_of_home_only=out_of_home_only,
        )
