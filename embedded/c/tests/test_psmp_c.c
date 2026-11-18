/**
 * @file test_psmp_c.c
 * @brief Unit tests and latency micro-benchmark for PSMP C Engine on QC8255.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "psmp_edge_engine.h"

static void test_table_initialization(void) {
    printf("[TEST] test_table_initialization ... ");
    psmp_ecdf_table_t table;
    uint32_t sample_durations[] = {10, 25, 45, 60, 120, 180, 480, 720};
    uint16_t count = sizeof(sample_durations) / sizeof(sample_durations[0]);

    int rc = psmp_ecdf_table_init(&table, sample_durations, count);
    assert(rc == 0);
    assert(table.count == count);
    assert(table.durations_min[0] == 10);
    assert(table.durations_min[count - 1] == 720);
    assert(table.cdf_q15[count - 1] == 32767); /* 最大確率 */

    /* 非ソート配列のエラー確認 */
    uint32_t unsorted[] = {10, 50, 20};
    assert(psmp_ecdf_table_init(&table, unsorted, 3) == -1);

    printf("PASSED\n");
}

static void test_hazard_calculation(void) {
    printf("[TEST] test_hazard_calculation ... ");
    psmp_ecdf_table_t table;
    /* 10分〜120分に分布 */
    uint32_t sample_durations[] = {15, 30, 45, 60, 75, 90, 105, 120};
    psmp_ecdf_table_init(&table, sample_durations, 8);

    /* 経過時間 0分、窓60分 -> 15, 30, 45, 60 の4件が窓内 -> 4/8 = 50% = 16384 */
    uint16_t h0 = psmp_calculate_hazard_q15(&table, 0, 60);
    assert(h0 == 16384);

    /* 経過時間 60分、窓60分 -> 生存件数 75, 90, 105, 120 (4件)。全て120分以下なので4件中4件 -> 100% = 32767 */
    uint16_t h60 = psmp_calculate_hazard_q15(&table, 60, 60);
    assert(h60 == 32767);

    /* 経過時間 150分 -> 全件超過 -> 尾部安全側で 32767 */
    uint16_t h150 = psmp_calculate_hazard_q15(&table, 150, 60);
    assert(h150 == 32767);

    printf("PASSED\n");
}

static void test_state_transitions_and_guards(void) {
    printf("[TEST] test_state_transitions_and_guards ... ");
    psmp_ecdf_table_t table;
    uint32_t sample_durations[] = {30, 60, 90, 120, 240, 480};
    psmp_ecdf_table_init(&table, sample_durations, 6);
    uint16_t threshold_q15 = 3277; /* 10% */

    psmp_parking_context_t ctx;
    psmp_decision_result_t res;

    /* 1. SoC Critical Guard -> S4 Dormant */
    ctx = (psmp_parking_context_t){
        .elapsed_min = 10,
        .category = PSMP_CAT_QUICK,
        .current_soc_pct = 15, /* <= 20% */
        .ambient_temp_c = 25,
        .is_home = false,
        .is_spontaneous_slot = false,
        .minutes_since_wake = 0,
        .door_unlocked = false
    };
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S4_DORMANT);

    /* 2. SoC Low Guard -> S3 DeepSleep Guarded */
    ctx.current_soc_pct = 25; /* <= 30% */
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S3_DEEPSLEEP_GUARDED);

    /* 3. Re-sleep Guard (15分放置で再退避) */
    ctx.current_soc_pct = 80;
    ctx.minutes_since_wake = 16; /* >= 15分 */
    ctx.door_unlocked = false;
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S2_DEEPSLEEP);

    /* 4. 出先短時間 チャタリング防止 (<= 25分) -> S0 Active Ready */
    ctx.minutes_since_wake = 0;
    ctx.category = PSMP_CAT_QUICK;
    ctx.elapsed_min = 15;
    ctx.is_home = false;
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S0_ACTIVE_READY);

    /* 5. 自宅長期放置 常用スロット -> S0 Active Ready */
    ctx.is_home = true;
    ctx.category = PSMP_CAT_HOME_LONG;
    ctx.elapsed_min = 3000;
    ctx.is_spontaneous_slot = false;
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S0_ACTIVE_READY);

    /* 6. 自宅長期放置 突発・非常用スロット -> S2 DeepSleep */
    ctx.is_spontaneous_slot = true;
    psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    assert(res.recommended_state == PSMP_STATE_S2_DEEPSLEEP);

    printf("PASSED\n");
}

static void benchmark_inference_latency(void) {
    printf("[BENCHMARK] QC8255 Cortex-R52 Inference Latency Benchmark ...\n");
    psmp_ecdf_table_t table;
    uint32_t durations[100];
    for (int i = 0; i < 100; i++) {
        durations[i] = (i + 1) * 10;
    }
    psmp_ecdf_table_init(&table, durations, 100);

    psmp_parking_context_t ctx = {
        .elapsed_min = 150,
        .category = PSMP_CAT_MEDIUM,
        .current_soc_pct = 85,
        .ambient_temp_c = 20,
        .is_home = false,
        .is_spontaneous_slot = false,
        .minutes_since_wake = 0,
        .door_unlocked = false
    };
    psmp_decision_result_t res;
    uint16_t threshold_q15 = 3277;

    const int ITERATIONS = 1000000;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < ITERATIONS; i++) {
        ctx.elapsed_min = (uint32_t)((i % 500) + 1);
        psmp_evaluate_state_transition(&ctx, &table, threshold_q15, &res);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    double avg_ns = (elapsed_sec / ITERATIONS) * 1e9;
    double avg_us = avg_ns / 1000.0;

    printf("  Total iterations : %d\n", ITERATIONS);
    printf("  Total time       : %.4f sec\n", elapsed_sec);
    printf("  Average latency  : %.2f ns (%.4f μs) per inference\n", avg_ns, avg_us);
    printf("  ROM / Footprint  : ECDF struct = %lu bytes\n", sizeof(psmp_ecdf_table_t));
    assert(avg_us < 5.0); /* 5μs未満の厳格基準クリア */
}

int main(void) {
    printf("==================================================================\n");
    printf("  PSMP C Engine Unit Tests & Benchmarks (Qualcomm QC8255 / R52)\n");
    printf("==================================================================\n");
    test_table_initialization();
    test_hazard_calculation();
    test_state_transitions_and_guards();
    benchmark_inference_latency();
    printf("==================================================================\n");
    printf("  ALL TESTS & BENCHMARKS PASSED (QC8255 Cortex-R52 Ready)\n");
    printf("==================================================================\n");
    return 0;
}
