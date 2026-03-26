/*
 * Test harness template for CUDA kernel evaluation.
 *
 * This file provides:
 *   1. Random input generation
 *   2. Reference CPU implementation for correctness verification
 *   3. Timing infrastructure for performance benchmarking
 *
 * The agent's kernel is expected to be in kernel.cu and must define:
 *   __global__ void target_kernel(const float* input, float* output, int M, int N)
 *
 * Compilation:
 *   nvcc -O2 -std=c++17 -o kernel_test kernel.cu test_harness.cu
 *
 * Usage:
 *   ./kernel_test <M> <N> [dtype]          — run correctness test
 *   ./kernel_test --benchmark <M> <N> [repeats] — run benchmark
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// Forward declaration — implemented by the agent in kernel.cu
extern __global__ void target_kernel(const float* input, float* output, int M, int N);

// Optional: agent may also provide a launch wrapper
// If not defined, the harness uses a default launch config
extern "C" {
    void launch_kernel(const float* d_input, float* d_output, int M, int N) __attribute__((weak));
}

// ---------------------------------------------------------------------------
// Reference CPU implementation (softmax by default — override via #define)
// ---------------------------------------------------------------------------
#ifndef REFERENCE_IMPL
static void reference_impl(const float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        float max_val = input[i * N];
        for (int j = 1; j < N; j++) {
            max_val = fmaxf(max_val, input[i * N + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            output[i * N + j] = expf(input[i * N + j] - max_val);
            sum += output[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            output[i * N + j] /= sum;
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Default kernel launcher
// ---------------------------------------------------------------------------
static void default_launch(const float* d_input, float* d_output, int M, int N) {
    dim3 block(256);
    dim3 grid(M);
    target_kernel<<<grid, block>>>(d_input, d_output, M, N);
}

static void do_launch(const float* d_input, float* d_output, int M, int N) {
    if (launch_kernel) {
        launch_kernel(d_input, d_output, M, N);
    } else {
        default_launch(d_input, d_output, M, N);
    }
}

// ---------------------------------------------------------------------------
// Correctness test
// ---------------------------------------------------------------------------
static bool run_correctness_test(int M, int N, float tolerance = 1e-5f) {
    size_t size = (size_t)M * N * sizeof(float);

    // Host buffers
    float* h_input  = (float*)malloc(size);
    float* h_ref    = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Random input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < M * N; i++) {
        h_input[i] = dist(rng);
    }

    // Reference
    reference_impl(h_input, h_ref, M, N);

    // Device buffers
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    do_launch(d_input, d_output, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Compare
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_output[i] - h_ref[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > tolerance) {
            if (fail_count < 5) {
                fprintf(stderr, "  Mismatch at [%d/%d, %d/%d]: got %.6f, expected %.6f (diff=%.6e)\n",
                        i / N, M, i % N, N, h_output[i], h_ref[i], diff);
            }
            fail_count++;
        }
    }

    // Cleanup
    free(h_input); free(h_ref); free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    if (fail_count > 0) {
        fprintf(stderr, "FAIL: %d/%d elements differ (max_diff=%.6e, tolerance=%.6e)\n",
                fail_count, M * N, max_diff, tolerance);
        printf("FAIL\n");
        return false;
    }

    printf("PASS (max_diff=%.6e)\n", max_diff);
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
static void run_benchmark(int M, int N, int repeats) {
    size_t size = (size_t)M * N * sizeof(float);

    float* h_input = (float*)malloc(size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < M * N; i++) h_input[i] = dist(rng);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; i++) {
        do_launch(d_input, d_output, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        do_launch(d_input, d_output, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / repeats;

    printf("TIME_MS: %.4f\n", avg_ms);
    printf("REPEATS: %d\n", repeats);
    printf("TOTAL_MS: %.4f\n", ms);

    // Cleanup
    free(h_input);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s <M> <N> [dtype]         — correctness test\n", argv[0]);
        fprintf(stderr, "  %s --benchmark <M> <N> [R] — benchmark\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--benchmark") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s --benchmark <M> <N> [repeats]\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]);
        int N = atoi(argv[3]);
        int repeats = (argc >= 5) ? atoi(argv[4]) : 5;
        run_benchmark(M, N, repeats);
        return 0;
    }

    int M = atoi(argv[1]);
    int N = (argc >= 3) ? atoi(argv[2]) : M;
    bool ok = run_correctness_test(M, N);
    return ok ? 0 : 1;
}
