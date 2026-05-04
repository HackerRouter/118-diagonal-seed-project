#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <cstring>

#define SILVER  0x6A09E667F3BCC909ULL
#define GOLDEN  0x9E3779B97F4A7C15ULL
#define SMUL1   0xBF58476D1CE4E5B9ULL
#define SMUL2   0x94D049BB133111EBULL
#define MASK60  ((1ULL << 60) - 1)

#define LCG_A   25214903917ULL
#define LCG_C   11ULL
#define LCG_M   ((1ULL << 48) - 1)

#define CHAIN_BATCH 8


__device__ __forceinline__ uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__ uint64_t mix_stafford13(uint64_t z) {
    z = (z ^ (z >> 30)) * SMUL1;
    z = (z ^ (z >> 27)) * SMUL2;
    return z ^ (z >> 31);
}

__device__ __forceinline__ void xoro_advance(uint64_t &s0, uint64_t &s1) {
    uint64_t t = s1 ^ s0;
    s0 = rotl64(s0, 49) ^ t ^ (t << 21);
    s1 = rotl64(t, 28);

__device__ __forceinline__ uint64_t xoro_next(uint64_t &s0, uint64_t &s1) {
    uint64_t r = rotl64(s0 + s1, 17) + s0;
    xoro_advance(s0, s1);
    return r;
}

__device__ __forceinline__ bool check_lower32(uint64_t s0, uint64_t s1) {
    xoro_advance(s0, s1);
    uint32_t al = (uint32_t)(xoro_next(s0, s1) >> 32) | 1U;

    xoro_advance(s0, s1);
    uint32_t bl = (uint32_t)(xoro_next(s0, s1) >> 32) | 1U;
    return (al == bl) || (al + bl == 0U);
}

__device__ __forceinline__ int check_full(uint64_t s0, uint64_t s1) {
    uint64_t r;

    r = xoro_next(s0, s1);
    uint32_t upper_a = (uint32_t)(r >> 32);

    r = xoro_next(s0, s1);
    uint32_t lower_a = (uint32_t)(r >> 32);

    r = xoro_next(s0, s1);
    uint32_t upper_b = (uint32_t)(r >> 32);

    r = xoro_next(s0, s1);
    uint32_t lower_b = (uint32_t)(r >> 32);

    uint64_t a = ((uint64_t)upper_a << 32) + (uint64_t)(int64_t)(int32_t)lower_a;
    uint64_t b = ((uint64_t)upper_b << 32) + (uint64_t)(int64_t)(int32_t)lower_b;
    a |= 1ULL; b |= 1ULL;

    return ((a + b) & MASK60) == 0 | ((a - b) & MASK60) == 0;
}

__global__ void search_range_kernel(uint64_t x_base_start, uint64_t total_seeds,
                                    uint64_t *results, uint32_t *result_count,
                                    uint32_t max_results) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t x_base = x_base_start + tid * (uint64_t)CHAIN_BATCH * GOLDEN;

    if (tid * CHAIN_BATCH >= total_seeds) return;

    uint64_t hashes[CHAIN_BATCH + 1];
    #pragma unroll
    for (int i = 0; i <= CHAIN_BATCH; i++)
        hashes[i] = mix_stafford13(x_base + (uint64_t)i * GOLDEN);

    #pragma unroll
    for (int i = 0; i < CHAIN_BATCH; i++) {
        uint64_t idx = tid * CHAIN_BATCH + i;
        if (idx >= total_seeds) return;

        uint64_t s0 = hashes[i];
        uint64_t s1 = hashes[i + 1];

        if (__builtin_expect((s0 | s1) == 0, 0)) {
            s0 = GOLDEN; s1 = SILVER;
        }
        if (__builtin_expect(check_lower32(s0, s1), 0)) {
            if (check_full(s0, s1)) {
                uint64_t world_seed = (x_base + (uint64_t)i * GOLDEN) ^ SILVER;
                uint32_t slot = atomicAdd(result_count, 1);
                if (slot < max_results) results[slot] = world_seed;
            }
        }
    }
}

__device__ __forceinline__ uint64_t lcg_state_to_world_seed(uint64_t state0) {
    uint64_t state1 = (state0 * LCG_A + LCG_C) & LCG_M;
    uint64_t state2 = (state1 * LCG_A + LCG_C) & LCG_M;
    uint32_t upper = (uint32_t)(state1 >> 16);
    uint32_t lower = (uint32_t)(state2 >> 16);
    return ((uint64_t)upper << 32) + (uint64_t)(int64_t)(int32_t)lower;
}

__global__ void search_lcg_kernel(uint64_t state_base, uint64_t total_states,
                                  uint64_t *results, uint32_t *result_count,
                                  uint32_t max_results) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < CHAIN_BATCH; i++) {
        uint64_t idx = tid * CHAIN_BATCH + i;
        if (idx >= total_states) return;

        uint64_t state = state_base + idx;
        uint64_t ws = lcg_state_to_world_seed(state);

        uint64_t lo_u = ws ^ SILVER;
        uint64_t hi_u = lo_u + GOLDEN;
        uint64_t s0 = mix_stafford13(lo_u);
        uint64_t s1 = mix_stafford13(hi_u);

        if (__builtin_expect((s0 | s1) == 0, 0)) {
            s0 = GOLDEN; s1 = SILVER;
        }

        if (__builtin_expect(check_lower32(s0, s1), 0)) {
            if (check_full(s0, s1)) {
                uint32_t slot = atomicAdd(result_count, 1);
                if (slot < max_results) results[slot] = ws;
            }
        }
    }
}

static uint64_t host_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t host_stafford13(uint64_t z) {
    z = (z ^ (z >> 30)) * SMUL1;
    z = (z ^ (z >> 27)) * SMUL2;
    return z ^ (z >> 31);
}

void host_compute_ab(uint64_t ws, uint64_t &a_out, uint64_t &b_out) {
    uint64_t lo_u = ws ^ SILVER;
    uint64_t hi_u = lo_u + GOLDEN;
    uint64_t s0 = host_stafford13(lo_u);
    uint64_t s1 = host_stafford13(hi_u);
    if ((s0 | s1) == 0) { s0 = GOLDEN; s1 = SILVER; }

    uint64_t r, t;
    r = host_rotl64(s0 + s1, 17) + s0;
    uint32_t ua = (uint32_t)(r >> 32);
    t = s1 ^ s0; s0 = host_rotl64(s0, 49) ^ t ^ (t << 21); s1 = host_rotl64(t, 28);

    r = host_rotl64(s0 + s1, 17) + s0;
    uint32_t la = (uint32_t)(r >> 32);
    t = s1 ^ s0; s0 = host_rotl64(s0, 49) ^ t ^ (t << 21); s1 = host_rotl64(t, 28);

    r = host_rotl64(s0 + s1, 17) + s0;
    uint32_t ub = (uint32_t)(r >> 32);
    t = s1 ^ s0; s0 = host_rotl64(s0, 49) ^ t ^ (t << 21); s1 = host_rotl64(t, 28);

    r = host_rotl64(s0 + s1, 17) + s0;
    uint32_t lb = (uint32_t)(r >> 32);

    a_out = (((uint64_t)ua << 32) + (uint64_t)(int64_t)(int32_t)la) | 1ULL;
    b_out = (((uint64_t)ub << 32) + (uint64_t)(int64_t)(int32_t)lb) | 1ULL;
}

void print_result(uint64_t seed, const char *mode_label, FILE *result_file) {
    uint64_t a, b;
    host_compute_ab(seed, a, b);
    uint64_t diff_low = (a - b) & MASK60;
    const char *dir = (diff_low == 0) ? "(-1,1)" : "(1,1)";

    printf("FOUND [%s] seed_unsigned=%" PRIu64 " seed_signed=%" PRId64
           " dir=%s a=0x%016" PRIX64 " b=0x%016" PRIX64 "\n",
           mode_label, seed, (int64_t)seed, dir, a, b);
    fflush(stdout);

    if (result_file) {
        fprintf(result_file,
                "seed_signed=%" PRId64 " seed_unsigned=%" PRIu64
                " dir=%s a=0x%016" PRIX64 " b=0x%016" PRIX64 " mode=%s\n",
                (int64_t)seed, seed, dir, a, b, mode_label);
        fflush(result_file);
    }
}

typedef void (*kernel_fn_t)(uint64_t, uint64_t, uint64_t*, uint32_t*, uint32_t);

void run_search(kernel_fn_t kernel, const char *mode_label,
                uint64_t start, uint64_t count) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("=== Diagonal Seed Finder for MC 26.1 (CUDA v3) ===\n");
    printf("Mode: %s\n", mode_label);
    printf("Search range: [%" PRIu64 ", %" PRIu64 ")\n", start, start + count);
    printf("Total to check: %" PRIu64 " (2^%.1f)\n", count, log2((double)count));
    printf("Chain batch: %d (chained hash + lower-32 filter)\n", CHAIN_BATCH);
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Results will be saved to: result.txt\n\n");

    FILE *result_file = fopen("result.txt", "a");
    if (!result_file)
        fprintf(stderr, "WARNING: could not open result.txt for writing\n");

    const uint32_t MAX_RESULTS = 1024;
    const int TPB = 256;

    uint64_t threads_per_batch = (uint64_t)TPB * prop.multiProcessorCount * 1024;
    uint64_t items_per_batch = threads_per_batch * CHAIN_BATCH;

    uint64_t *d_results;
    uint32_t *d_count;
    cudaMalloc(&d_results, MAX_RESULTS * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    uint64_t *h_results = (uint64_t *)malloc(MAX_RESULTS * sizeof(uint64_t));
    uint64_t total_found = 0;
    uint64_t done = 0;

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);
    float last_report_ms = 0;

    for (uint64_t offset = 0; offset < count; offset += items_per_batch) {
        uint64_t batch = items_per_batch;
        if (offset + batch > count) batch = count - offset;

        uint64_t batch_threads = (batch + CHAIN_BATCH - 1) / CHAIN_BATCH;
        int blocks = (int)((batch_threads + TPB - 1) / TPB);

        kernel<<<blocks, TPB>>>(start + offset, batch, d_results, d_count, MAX_RESULTS);

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end);
        done += batch;

        if (elapsed_ms - last_report_ms > 2000.0f || done >= count) {
            last_report_ms = elapsed_ms;

            uint32_t h_count;
            cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (h_count > total_found) {
                uint32_t n = h_count - (uint32_t)total_found;
                uint32_t s = (uint32_t)total_found;
                uint32_t c = (n <= MAX_RESULTS - s) ? n : MAX_RESULTS - s;
                if (c > 0) {
                    cudaMemcpy(h_results, d_results + s, c * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost);
                    for (uint32_t i = 0; i < c; i++)
                        print_result(h_results[i], mode_label, result_file);
                }
                total_found = h_count;
            }

            double sec = elapsed_ms / 1000.0;
            double rate = done / sec;
            double pct = 100.0 * done / count;
            double eta = (count - done) / rate;
            fprintf(stderr, "\r[%.1f%%] %" PRIu64 "/%" PRIu64 " | %.3e/s | "
                    "found: %" PRIu64 " | %.0fs elapsed | ETA: %.0fs   ",
                    pct, done, count, rate, total_found, sec, eta);
        }
    }

    fprintf(stderr, "\n");
    printf("\n=== Done. Found %" PRIu64 " diagonal seed(s). ===\n", total_found);

    if (result_file) fclose(result_file);
    cudaFree(d_results);
    cudaFree(d_count);
    free(h_results);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

void print_usage() {
    fprintf(stderr,
        "Usage:\n"
        "  diagonal_finder_v3                          range mode, 0..2^48\n"
        "  diagonal_finder_v3 range [start] [count]    range mode, explicit\n"
        "  diagonal_finder_v3 lcg                      LCG mode, all 2^48 states\n"
        "  diagonal_finder_v3 lcg [start] [count]      LCG mode, explicit\n");
}

int main(int argc, char **argv) {
    const char *mode = "range";
    uint64_t start = 0;
    uint64_t count = (1ULL << 48);
    int arg_offset = 1;

    if (argc > 1) {
        if (strcmp(argv[1], "range") == 0 || strcmp(argv[1], "lcg") == 0) {
            mode = argv[1];
            arg_offset = 2;
        } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    if (argc > arg_offset + 1) {
        start = strtoull(argv[arg_offset], NULL, 0);
        count = strtoull(argv[arg_offset + 1], NULL, 0);
    } else if (argc > arg_offset) {
        count = strtoull(argv[arg_offset], NULL, 0);
    }

    if (strcmp(mode, "lcg") == 0 && start + count > (1ULL << 48)) {
        count = (1ULL << 48) - start;
        fprintf(stderr, "Note: LCG state space is 2^48, capping count to %" PRIu64 "\n", count);
    }

    if (strcmp(mode, "lcg") == 0) {
        run_search(search_lcg_kernel, "lcg", start, count);
    } else {
        run_search(search_range_kernel, "range", start, count);
    }

    return 0;
}
