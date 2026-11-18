/**
 * @file psmp_edge_engine.h
 * @brief PSMP (Parking State Management & Prediction) Edge Engine for Qualcomm QC8255.
 * 
 * Designed for Qualcomm SA8255P Low-Power Real-Time Subsystem (ARM Cortex-R52 / AON MCU).
 * - Zero dynamic memory allocation (no malloc/free)
 * - Deterministic O(log N) binary search inference
 * - MISRA-C / ISO 26262 ASIL-B/D compliant design
 * - Fixed-point / Q15 probability representation option for ultra-low power execution
 */

#ifndef PSMP_EDGE_ENGINE_H
#define PSMP_EDGE_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define PSMP_MAX_ECDF_POINTS 128
#define PSMP_QUICK_STAY_THRESHOLD_MIN 25
#define PSMP_RESLEEP_GUARD_TIMEOUT_MIN 15
#define PSMP_SOC_CRITICAL_THRESHOLD_PCT 20
#define PSMP_SOC_LOW_THRESHOLD_PCT 30

/**
 * @brief 車両省電力状態（4段階＋休眠）
 */
typedef enum {
    PSMP_STATE_S0_ACTIVE_READY      = 0,  /**< 完全待受 (暗電流 ~35mA, 即時解錠) */
    PSMP_STATE_S1_LIGHT_STANDBY     = 1,  /**< 軽待受 (暗電流 ~15mA) */
    PSMP_STATE_S2_DEEPSLEEP         = 2,  /**< ディープスリープ (暗電流 ~3.5mA, AP電源遮断) */
    PSMP_STATE_S3_DEEPSLEEP_GUARDED = 3,  /**< ガード付きディープスリープ (タイマー監視) */
    PSMP_STATE_S4_DORMANT           = 4   /**< 完全休眠 / 長期保管 (暗電流 < 1mA) */
} psmp_power_state_t;

/**
 * @brief 滞在カテゴリ種別
 */
typedef enum {
    PSMP_CAT_HOME_OVERNIGHT = 0, /**< 自宅夜間 (<= 24h) */
    PSMP_CAT_HOME_LONG      = 1, /**< 自宅長期放置 (> 24h, 3日以上目標) */
    PSMP_CAT_QUICK          = 2, /**< 出先短時間 (コンビニ・送迎等, <= 25分) */
    PSMP_CAT_MEDIUM         = 3, /**< 出先中時間 (買物・食事等, 25分〜120分) */
    PSMP_CAT_LONG           = 4  /**< 出先長時間 (観光・勤務等, > 120分) */
} psmp_poi_category_t;

/**
 * @brief ソート済みECDF分布テーブル (TCM / ROM / SRAM展開用)
 */
typedef struct {
    uint16_t count;                                  /**< 有効ポイント数 (<= PSMP_MAX_ECDF_POINTS) */
    uint32_t durations_min[PSMP_MAX_ECDF_POINTS];     /**< 滞在時間 [分] (昇順ソート済み) */
    uint16_t cdf_q15[PSMP_MAX_ECDF_POINTS];          /**< 累積確率 F(t) [Q15固定小数点: 0..32767] */
} psmp_ecdf_table_t;

/**
 * @brief 駐車コンテキスト構造体
 */
typedef struct {
    uint32_t elapsed_min;             /**< 駐車開始からの経過時間 [分] */
    psmp_poi_category_t category;     /**< 滞在カテゴリ */
    uint8_t current_soc_pct;          /**< 補機/主機バッテリSoC [%] */
    int8_t ambient_temp_c;            /**< 外気温度 [℃] */
    bool is_home;                     /**< 自宅フラグ */
    bool is_spontaneous_slot;         /**< 過去乗車実績のない突発時間帯フラグ */
    uint16_t minutes_since_wake;      /**< 待受起床後の経過時間 [分] (Re-sleep Guard判定用) */
    bool door_unlocked;               /**< ドア解錠フラグ */
} psmp_parking_context_t;

/**
 * @brief 決定結果構造体
 */
typedef struct {
    psmp_power_state_t recommended_state; /**< 推奨電力状態 */
    uint16_t hazard_rate_q15;             /**< 次窓乗車ハザード確率 [Q15: 0..32767] */
    bool wake_timer_active;               /**< タイマー復帰予約フラグ */
    uint32_t next_wake_min;               /**< 次回復帰予定時間 [分] */
    const char *decision_reason;          /**< 判定理由文字列 (デバッグ/ログ用) */
} psmp_decision_result_t;

/**
 * @brief ECDFテーブルの初期化・検証
 * @param table 初期化対象テーブルポインタ
 * @param durations_min 昇順ソート済み滞在時間配列 [分]
 * @param count データ数
 * @return 0: 成功, -1: 引数不正または非ソート
 */
int psmp_ecdf_table_init(psmp_ecdf_table_t *table, const uint32_t *durations_min, uint16_t count);

/**
 * @brief 経過時間 t におけるハザード確率 P(乗車 | t経過) の二分探索推論
 * @param table ECDFテーブル
 * @param elapsed_min 経過時間 [分]
 * @param window_min 予測対象窓幅 [分] (既定: 60分)
 * @return ハザード確率 [Q15: 0..32767, 32768=1.0]
 */
uint16_t psmp_calculate_hazard_q15(const psmp_ecdf_table_t *table, uint32_t elapsed_min, uint32_t window_min);

/**
 * @brief QC8255 Cortex-R52 向け省電力状態判定エントリポイント
 * @param ctx 駐車コンテキスト
 * @param table 対応ECDFテーブル (自宅/出先)
 * @param threshold_q15 復帰ハザード閾値 (既定: 0.10 * 32768 ≒ 3277)
 * @param result 決定結果出力先
 * @return 0: 正常終了, 負値: エラー
 */
int psmp_evaluate_state_transition(
    const psmp_parking_context_t *ctx,
    const psmp_ecdf_table_t *table,
    uint16_t threshold_q15,
    psmp_decision_result_t *result
);

#ifdef __cplusplus
}
#endif

#endif /* PSMP_EDGE_ENGINE_H */
