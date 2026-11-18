/**
 * @file psmp_edge_engine.c
 * @brief PSMP Edge Engine Implementation for Qualcomm QC8255 (SA8255P).
 */

#include "psmp_edge_engine.h"

int psmp_ecdf_table_init(psmp_ecdf_table_t *table, const uint32_t *durations_min, uint16_t count) {
    if (!table || !durations_min || count == 0 || count > PSMP_MAX_ECDF_POINTS) {
        return -1;
    }

    table->count = count;
    for (uint16_t i = 0; i < count; i++) {
        if (i > 0 && durations_min[i] < durations_min[i - 1]) {
            return -1; /* 非ソート配列 */
        }
        table->durations_min[i] = durations_min[i];
        /* Q15 CDF: (i + 1) / count * 32768 */
        uint32_t cdf_val = ((uint32_t)(i + 1) * 32768U) / (uint32_t)count;
        if (cdf_val > 32767U) {
            cdf_val = 32767U;
        }
        table->cdf_q15[i] = (uint16_t)cdf_val;
    }
    return 0;
}

/**
 * @brief 二分探索で target 以下の要素数をカウント (upper_bound)
 */
static uint16_t binary_search_upper_bound(const uint32_t *arr, uint16_t n, uint32_t target) {
    uint16_t low = 0;
    uint16_t high = n;

    while (low < high) {
        uint16_t mid = low + (high - low) / 2;
        if (arr[mid] <= target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

uint16_t psmp_calculate_hazard_q15(const psmp_ecdf_table_t *table, uint32_t elapsed_min, uint32_t window_min) {
    if (!table || table->count == 0) {
        return 0;
    }

    uint16_t count = table->count;
    /* 経過時間 t までに既に離脱（乗車）した件数 */
    uint16_t idx_t = binary_search_upper_bound(table->durations_min, count, elapsed_min);

    /* 生存件数 S(t) = count - idx_t */
    if (idx_t >= count) {
        /* 全ての過去実績時間を超過している場合: 尾部ハザード（安全側で復帰） */
        return 32767U;
    }

    uint32_t survival_count = (uint32_t)(count - idx_t);
    uint32_t target_window = elapsed_min + window_min;
    uint16_t idx_w = binary_search_upper_bound(table->durations_min, count, target_window);

    /* 窓内乗車イベント数 = idx_w - idx_t */
    uint32_t events_in_window = (uint32_t)(idx_w - idx_t);

    /* ハザード率 H = events_in_window / survival_count in Q15 */
    uint32_t hazard_q15 = (events_in_window * 32768U) / survival_count;
    if (hazard_q15 > 32767U) {
        hazard_q15 = 32767U;
    }

    return (uint16_t)hazard_q15;
}

int psmp_evaluate_state_transition(
    const psmp_parking_context_t *ctx,
    const psmp_ecdf_table_t *table,
    uint16_t threshold_q15,
    psmp_decision_result_t *result
) {
    if (!ctx || !result) {
        return -1;
    }

    /* 既定値初期化 */
    result->recommended_state = PSMP_STATE_S2_DEEPSLEEP;
    result->hazard_rate_q15 = 0;
    result->wake_timer_active = false;
    result->next_wake_min = 0;
    result->decision_reason = "DEFAULT_DEEPSLEEP";

    /* ==========================================================
     * Guard 1: SoC 安全弁 (バッテリ上がり完全防止)
     * ========================================================== */
    if (ctx->current_soc_pct <= PSMP_SOC_CRITICAL_THRESHOLD_PCT) {
        result->recommended_state = PSMP_STATE_S4_DORMANT;
        result->decision_reason = "GUARD_CRITICAL_SOC_DORMANT";
        return 0;
    }
    if (ctx->current_soc_pct <= PSMP_SOC_LOW_THRESHOLD_PCT) {
        result->recommended_state = PSMP_STATE_S3_DEEPSLEEP_GUARDED;
        result->decision_reason = "GUARD_LOW_SOC_ENFORCED_DEEPSLEEP";
        return 0;
    }

    /* ==========================================================
     * Guard 2: Re-sleep 安全弁 (予定変更・空振り防止)
     * ========================================================== */
    if (!ctx->door_unlocked && ctx->minutes_since_wake >= PSMP_RESLEEP_GUARD_TIMEOUT_MIN) {
        result->recommended_state = PSMP_STATE_S2_DEEPSLEEP;
        result->decision_reason = "GUARD_RESLEEP_TIMEOUT_RETREAT";
        return 0;
    }

    /* ==========================================================
     * Case 1: 出先短時間 (チャタリング防止)
     * ========================================================== */
    if (!ctx->is_home && ctx->category == PSMP_CAT_QUICK && ctx->elapsed_min <= PSMP_QUICK_STAY_THRESHOLD_MIN) {
        result->recommended_state = PSMP_STATE_S0_ACTIVE_READY;
        result->decision_reason = "OUT_QUICK_STAY_CHATTERING_PREVENTION";
        return 0;
    }

    /* ==========================================================
     * Case 2: 自宅長期放置 (HOME_LONG, 3日以上放置ターゲット)
     * ========================================================== */
    if (ctx->is_home && ctx->category == PSMP_CAT_HOME_LONG) {
        if (ctx->is_spontaneous_slot) {
            /* 過去実績のない時間帯はDeepSleep固定 */
            result->recommended_state = PSMP_STATE_S2_DEEPSLEEP;
            result->decision_reason = "HOME_LONG_NON_SLOT_DEEPSLEEP";
        } else {
            /* 常用乗車スロット: 待受復帰 */
            result->recommended_state = PSMP_STATE_S0_ACTIVE_READY;
            result->decision_reason = "HOME_LONG_REGULAR_SLOT_WAKEUP";
        }
        return 0;
    }

    /* ==========================================================
     * Case 3: ECDFハザード率算出による自律推論 (出先中長時間・自宅夜間)
     * ========================================================== */
    if (table && table->count > 0) {
        result->hazard_rate_q15 = psmp_calculate_hazard_q15(table, ctx->elapsed_min, 60U);

        if (result->hazard_rate_q15 >= threshold_q15) {
            result->recommended_state = PSMP_STATE_S0_ACTIVE_READY;
            result->decision_reason = "HAZARD_THRESHOLD_EXCEEDED_WAKE";
        } else {
            result->recommended_state = PSMP_STATE_S2_DEEPSLEEP;
            result->wake_timer_active = true;
            result->next_wake_min = ctx->elapsed_min + 60U;
            result->decision_reason = "HAZARD_LOW_SUSTAIN_DEEPSLEEP";
        }
    } else {
        /* テーブルなし時フォールバック */
        result->recommended_state = PSMP_STATE_S1_LIGHT_STANDBY;
        result->decision_reason = "NO_TABLE_FALLBACK_STANDBY";
    }

    return 0;
}
