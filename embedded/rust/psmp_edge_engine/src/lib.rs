//! psmp-edge-engine
//!
//! Ultra-lightweight, zero-allocation `#![no_std]` parking state management and prediction
//! engine for Qualcomm QC8255 (SA8255P) Automotive Cockpit Platform (Cortex-R52 / Kryo).

#![cfg_attr(not(test), no_std)]

pub const MAX_ECDF_POINTS: usize = 128;
pub const QUICK_STAY_THRESHOLD_MIN: u32 = 25;
pub const RESLEEP_GUARD_TIMEOUT_MIN: u16 = 15;
pub const SOC_CRITICAL_THRESHOLD_PCT: u8 = 20;
pub const SOC_LOW_THRESHOLD_PCT: u8 = 30;

/// 車両省電力状態
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PowerState {
    S0ActiveReady = 0,
    S1LightStandby = 1,
    S2DeepSleep = 2,
    S3DeepSleepGuarded = 3,
    S4Dormant = 4,
}

/// 滞在カテゴリ種別
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PoiCategory {
    HomeOvernight = 0,
    HomeLong = 1,
    Quick = 2,
    Medium = 3,
    Long = 4,
}

/// ソート済みECDF分布テーブル (TCM / ROM / SRAM展開用)
#[derive(Debug, Clone, Copy)]
pub struct EcdfTable<const N: usize = MAX_ECDF_POINTS> {
    pub count: usize,
    pub durations_min: [u32; N],
    pub cdf_q15: [u16; N],
}

impl<const N: usize> Default for EcdfTable<N> {
    fn default() -> Self {
        Self {
            count: 0,
            durations_min: [0; N],
            cdf_q15: [0; N],
        }
    }
}

impl<const N: usize> EcdfTable<N> {
    pub const fn new() -> Self {
        Self {
            count: 0,
            durations_min: [0; N],
            cdf_q15: [0; N],
        }
    }

    /// 昇順ソート済みスライスからテーブルを初期化
    pub fn from_slice(slice: &[u32]) -> Result<Self, &'static str> {
        if slice.is_empty() || slice.len() > N {
            return Err("Invalid slice length");
        }

        let mut table = Self::new();
        table.count = slice.len();

        let mut prev = 0;
        for (i, &dur) in slice.iter().enumerate() {
            if i > 0 && dur < prev {
                return Err("Input slice must be sorted in ascending order");
            }
            prev = dur;
            table.durations_min[i] = dur;
            let cdf_val = ((i as u64 + 1) * 32768) / slice.len() as u64;
            table.cdf_q15[i] = if cdf_val > 32767 { 32767 } else { cdf_val as u16 };
        }
        Ok(table)
    }

    /// 経過時間 elapsed_min におけるハザード確率 P(乗車 | t経過) の二分探索推論
    pub fn calculate_hazard_q15(&self, elapsed_min: u32, window_min: u32) -> u16 {
        if self.count == 0 {
            return 0;
        }

        let active_slice = &self.durations_min[..self.count];
        let idx_t = active_slice.partition_point(|&x| x <= elapsed_min);

        if idx_t >= self.count {
            return 32767; // 尾部安全側で復帰
        }

        let survival_count = (self.count - idx_t) as u64;
        let target_window = elapsed_min.saturating_add(window_min);
        let idx_w = active_slice.partition_point(|&x| x <= target_window);

        let events_in_window = (idx_w - idx_t) as u64;
        let hazard_q15 = (events_in_window * 32768) / survival_count;

        if hazard_q15 > 32767 {
            32767
        } else {
            hazard_q15 as u16
        }
    }
}

/// 駐車コンテキスト
#[derive(Debug, Clone, Copy)]
pub struct ParkingContext {
    pub elapsed_min: u32,
    pub category: PoiCategory,
    pub current_soc_pct: u8,
    pub ambient_temp_c: i8,
    pub is_home: bool,
    pub is_spontaneous_slot: bool,
    pub minutes_since_wake: u16,
    pub door_unlocked: bool,
}

/// 決定結果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionResult {
    pub recommended_state: PowerState,
    pub hazard_rate_q15: u16,
    pub wake_timer_active: bool,
    pub next_wake_min: u32,
    pub decision_reason: &'static str,
}

/// QC8255 Cortex-R52 向け省電力状態判定エントリポイント
pub fn evaluate_state_transition<const N: usize>(
    ctx: &ParkingContext,
    table: Option<&EcdfTable<N>>,
    threshold_q15: u16,
) -> DecisionResult {
    // 1. SoC ガード (バッテリ上がり完全防止)
    if ctx.current_soc_pct <= SOC_CRITICAL_THRESHOLD_PCT {
        return DecisionResult {
            recommended_state: PowerState::S4Dormant,
            hazard_rate_q15: 0,
            wake_timer_active: false,
            next_wake_min: 0,
            decision_reason: "GUARD_CRITICAL_SOC_DORMANT",
        };
    }
    if ctx.current_soc_pct <= SOC_LOW_THRESHOLD_PCT {
        return DecisionResult {
            recommended_state: PowerState::S3DeepSleepGuarded,
            hazard_rate_q15: 0,
            wake_timer_active: false,
            next_wake_min: 0,
            decision_reason: "GUARD_LOW_SOC_ENFORCED_DEEPSLEEP",
        };
    }

    // 2. Re-sleep ガード (予定変更・空振り防止)
    if !ctx.door_unlocked && ctx.minutes_since_wake >= RESLEEP_GUARD_TIMEOUT_MIN {
        return DecisionResult {
            recommended_state: PowerState::S2DeepSleep,
            hazard_rate_q15: 0,
            wake_timer_active: false,
            next_wake_min: 0,
            decision_reason: "GUARD_RESLEEP_TIMEOUT_RETREAT",
        };
    }

    // 3. 出先短時間 チャタリング防止
    if !ctx.is_home && ctx.category == PoiCategory::Quick && ctx.elapsed_min <= QUICK_STAY_THRESHOLD_MIN {
        return DecisionResult {
            recommended_state: PowerState::S0ActiveReady,
            hazard_rate_q15: 0,
            wake_timer_active: false,
            next_wake_min: 0,
            decision_reason: "OUT_QUICK_STAY_CHATTERING_PREVENTION",
        };
    }

    // 4. 自宅長期放置 (HOME_LONG)
    if ctx.is_home && ctx.category == PoiCategory::HomeLong {
        return if ctx.is_spontaneous_slot {
            DecisionResult {
                recommended_state: PowerState::S2DeepSleep,
                hazard_rate_q15: 0,
                wake_timer_active: false,
                next_wake_min: 0,
                decision_reason: "HOME_LONG_NON_SLOT_DEEPSLEEP",
            }
        } else {
            DecisionResult {
                recommended_state: PowerState::S0ActiveReady,
                hazard_rate_q15: 0,
                wake_timer_active: false,
                next_wake_min: 0,
                decision_reason: "HOME_LONG_REGULAR_SLOT_WAKEUP",
            }
        };
    }

    // 5. ECDFハザード率算出による自律推論
    if let Some(t) = table {
        if t.count > 0 {
            let hazard_q15 = t.calculate_hazard_q15(ctx.elapsed_min, 60);
            return if hazard_q15 >= threshold_q15 {
                DecisionResult {
                    recommended_state: PowerState::S0ActiveReady,
                    hazard_rate_q15: hazard_q15,
                    wake_timer_active: false,
                    next_wake_min: 0,
                    decision_reason: "HAZARD_THRESHOLD_EXCEEDED_WAKE",
                }
            } else {
                DecisionResult {
                    recommended_state: PowerState::S2DeepSleep,
                    hazard_rate_q15: hazard_q15,
                    wake_timer_active: true,
                    next_wake_min: ctx.elapsed_min.saturating_add(60),
                    decision_reason: "HAZARD_LOW_SUSTAIN_DEEPSLEEP",
                }
            };
        }
    }

    // テーブルなし時フォールバック
    DecisionResult {
        recommended_state: PowerState::S1LightStandby,
        hazard_rate_q15: 0,
        wake_timer_active: false,
        next_wake_min: 0,
        decision_reason: "NO_TABLE_FALLBACK_STANDBY",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecdf_table_creation() {
        let durations = [10, 20, 30, 45, 60];
        let table = EcdfTable::<16>::from_slice(&durations).unwrap();
        assert_eq!(table.count, 5);
        assert_eq!(table.durations_min[0], 10);
        assert_eq!(table.durations_min[4], 60);

        let unsorted = [10, 30, 20];
        assert!(EcdfTable::<16>::from_slice(&unsorted).is_err());
    }

    #[test]
    fn test_hazard_calculation() {
        let durations = [15, 30, 45, 60, 75, 90, 105, 120];
        let table = EcdfTable::<16>::from_slice(&durations).unwrap();

        // t=0, window=60 -> 4/8 = 50% = 16384
        assert_eq!(table.calculate_hazard_q15(0, 60), 16384);
        // t=60, window=60 -> 4/4 = 100% = 32767
        assert_eq!(table.calculate_hazard_q15(60, 60), 32767);
        // t=150 -> 尾部安全側 = 32767
        assert_eq!(table.calculate_hazard_q15(150, 60), 32767);
    }

    #[test]
    fn test_guards_and_transitions() {
        let durations = [30, 60, 90, 120, 240, 480];
        let table = EcdfTable::<16>::from_slice(&durations).unwrap();
        let threshold_q15 = 3277; // 10%

        let mut ctx = ParkingContext {
            elapsed_min: 10,
            category: PoiCategory::Quick,
            current_soc_pct: 15, // <= 20%
            ambient_temp_c: 25,
            is_home: false,
            is_spontaneous_slot: false,
            minutes_since_wake: 0,
            door_unlocked: false,
        };

        // 1. Critical SoC
        let res = evaluate_state_transition(&ctx, Some(&table), threshold_q15);
        assert_eq!(res.recommended_state, PowerState::S4Dormant);

        // 2. Low SoC
        ctx.current_soc_pct = 25;
        let res = evaluate_state_transition(&ctx, Some(&table), threshold_q15);
        assert_eq!(res.recommended_state, PowerState::S3DeepSleepGuarded);

        // 3. Re-sleep Guard
        ctx.current_soc_pct = 80;
        ctx.minutes_since_wake = 16;
        let res = evaluate_state_transition(&ctx, Some(&table), threshold_q15);
        assert_eq!(res.recommended_state, PowerState::S2DeepSleep);

        // 4. Quick Stay
        ctx.minutes_since_wake = 0;
        ctx.elapsed_min = 15;
        let res = evaluate_state_transition(&ctx, Some(&table), threshold_q15);
        assert_eq!(res.recommended_state, PowerState::S0ActiveReady);
    }
}
