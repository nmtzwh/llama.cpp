#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)

static inline void simd_gemm_sve_4x2(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int j, int vl)
{
    svbool_t pg0 = svwhilelt_b32(j,      N);
    svbool_t pg1 = svwhilelt_b32(j + vl, N);

    svfloat32_t c00 = svld1_f32(pg0, C + 0 * N + j);
    svfloat32_t c01 = svld1_f32(pg1, C + 0 * N + j + vl);
    svfloat32_t c10 = svld1_f32(pg0, C + 1 * N + j);
    svfloat32_t c11 = svld1_f32(pg1, C + 1 * N + j + vl);
    svfloat32_t c20 = svld1_f32(pg0, C + 2 * N + j);
    svfloat32_t c21 = svld1_f32(pg1, C + 2 * N + j + vl);
    svfloat32_t c30 = svld1_f32(pg0, C + 3 * N + j);
    svfloat32_t c31 = svld1_f32(pg1, C + 3 * N + j + vl);

    for (int kk = 0; kk < K; kk++) {
        const float * b = B + kk * N + j;
        svfloat32_t b0 = svld1_f32(pg0, b);
        svfloat32_t b1 = svld1_f32(pg1, b + vl);

        svfloat32_t a0 = svdup_n_f32(A[0 * K + kk]);
        svfloat32_t a1 = svdup_n_f32(A[1 * K + kk]);
        svfloat32_t a2 = svdup_n_f32(A[2 * K + kk]);
        svfloat32_t a3 = svdup_n_f32(A[3 * K + kk]);

        c00 = svmla_f32_x(pg0, c00, b0, a0);
        c01 = svmla_f32_x(pg1, c01, b1, a0);
        c10 = svmla_f32_x(pg0, c10, b0, a1);
        c11 = svmla_f32_x(pg1, c11, b1, a1);
        c20 = svmla_f32_x(pg0, c20, b0, a2);
        c21 = svmla_f32_x(pg1, c21, b1, a2);
        c30 = svmla_f32_x(pg0, c30, b0, a3);
        c31 = svmla_f32_x(pg1, c31, b1, a3);
    }

    svst1_f32(pg0, C + 0 * N + j,      c00);
    svst1_f32(pg1, C + 0 * N + j + vl, c01);
    svst1_f32(pg0, C + 1 * N + j,      c10);
    svst1_f32(pg1, C + 1 * N + j + vl, c11);
    svst1_f32(pg0, C + 2 * N + j,      c20);
    svst1_f32(pg1, C + 2 * N + j + vl, c21);
    svst1_f32(pg0, C + 3 * N + j,      c30);
    svst1_f32(pg1, C + 3 * N + j + vl, c31);
}

static inline void simd_gemm_sve_4x1(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int j)
{
    svbool_t pg = svwhilelt_b32(j, N);

    svfloat32_t c0 = svld1_f32(pg, C + 0 * N + j);
    svfloat32_t c1 = svld1_f32(pg, C + 1 * N + j);
    svfloat32_t c2 = svld1_f32(pg, C + 2 * N + j);
    svfloat32_t c3 = svld1_f32(pg, C + 3 * N + j);

    for (int kk = 0; kk < K; kk++) {
        svfloat32_t b = svld1_f32(pg, B + kk * N + j);

        c0 = svmla_f32_x(pg, c0, b, svdup_n_f32(A[0 * K + kk]));
        c1 = svmla_f32_x(pg, c1, b, svdup_n_f32(A[1 * K + kk]));
        c2 = svmla_f32_x(pg, c2, b, svdup_n_f32(A[2 * K + kk]));
        c3 = svmla_f32_x(pg, c3, b, svdup_n_f32(A[3 * K + kk]));
    }

    svst1_f32(pg, C + 0 * N + j, c0);
    svst1_f32(pg, C + 1 * N + j, c1);
    svst1_f32(pg, C + 2 * N + j, c2);
    svst1_f32(pg, C + 3 * N + j, c3);
}

static inline void simd_gemm_sve_1x2(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int j, int vl)
{
    svbool_t pg0 = svwhilelt_b32(j,      N);
    svbool_t pg1 = svwhilelt_b32(j + vl, N);

    svfloat32_t c0 = svld1_f32(pg0, C + j);
    svfloat32_t c1 = svld1_f32(pg1, C + j + vl);

    for (int kk = 0; kk < K; kk++) {
        const float * b = B + kk * N + j;
        svfloat32_t a = svdup_n_f32(A[kk]);

        c0 = svmla_f32_x(pg0, c0, svld1_f32(pg0, b),      a);
        c1 = svmla_f32_x(pg1, c1, svld1_f32(pg1, b + vl), a);
    }

    svst1_f32(pg0, C + j,      c0);
    svst1_f32(pg1, C + j + vl, c1);
}

static inline void simd_gemm_sve_1x1(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int j)
{
    svbool_t pg = svwhilelt_b32(j, N);
    svfloat32_t c = svld1_f32(pg, C + j);

    for (int kk = 0; kk < K; kk++) {
        c = svmla_f32_x(pg, c, svld1_f32(pg, B + kk * N + j), svdup_n_f32(A[kk]));
    }

    svst1_f32(pg, C + j, c);
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int vl = svcntw();

    int i = 0;
    for (; i + 4 <= M; i += 4) {
        int j = 0;
        for (; j + vl < N; j += 2 * vl) {
            simd_gemm_sve_4x2(C, A, B, K, N, j, vl);
        }
        if (j < N) {
            simd_gemm_sve_4x1(C, A, B, K, N, j);
        }

        A += 4 * K;
        C += 4 * N;
    }

    for (; i < M; i++) {
        int j = 0;
        for (; j + vl < N; j += 2 * vl) {
            simd_gemm_sve_1x2(C, A, B, K, N, j, vl);
        }
        if (j < N) {
            simd_gemm_sve_1x1(C, A, B, K, N, j);
        }

        A += K;
        C += N;
    }
}

// TODO: add support for more sizeless vector types
#elif defined(GGML_SIMD) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            for (int64_t i = 0; i < GEMM_RM; i++) {
                float a = C[i * N + jj];
                for (int64_t kk = 0; kk < K; kk++) {
                    a += A[i + kk] * B[kk * N + jj];
                }
                C[i * N + jj] = a;
            }
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            float a = C[jj];
            for (int64_t kk = 0; kk < K; kk++) {
                a += A[kk] * B[kk * N + jj];
            }
            C[jj] = a;
        }

        A += K;
        C += N;
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
