#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>

static const uint64_t SILVER_RATIO = 0x6A09E667F3BCC909ULL;
static const uint64_t GOLDEN_RATIO = 0x9E3779B97F4A7C15ULL;
static const uint64_t STAFFORD_MUL1 = 0xBF58476D1CE4E5B9ULL;
static const uint64_t STAFFORD_MUL2 = 0x94D049BB133111EBULL;
static const uint64_t MASK60 = (1ULL << 60) - 1;

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__ uint64_t mix_stafford13(uint64_t z) {
    z = (z ^ (z >> 30)) * STAFFORD_MUL1;
    z = (z ^ (z >> 27)) * STAFFORD_MUL2;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t xoroshiro_next(uint64_t &s0, uint64_t &s1) {
    uint64_t result = rotl64(s0 + s1, 17) + s0;
    uint64_t t = s1 ^ s0;
    s0 = rotl64(s0, 49) ^ t ^ (t << 21);
    s1 = rotl64(t, 28);
    return result;
}

__device__ __forceinline__ uint64_t worldgen_next_long(uint64_t &s0, uint64_t &s1) {
    uint64_t r0 = xoroshiro_next(s0, s1);
    uint32_t upper = (uint32_t)(r0 >> 32);

    uint64_t r1 = xoroshiro_next(s0, s1);
    uint32_t lower = (uint32_t)(r1 >> 32);

    int64_t lower_ext = (int64_t)(int32_t)lower;
    uint64_t result = ((uint64_t)upper << 32) + (uint64_t)lower_ext;
    return result;
}

__global__ void search_kernel(uint64_t seed_start, uint64_t seed_count,
                              uint64_t *results, uint32_t *result_count,
                              uint32_t max_results) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count) return;

    uint64_t world_seed = seed_start + idx;

    uint64_t lo_unmixed = world_seed ^ SILVER_RATIO;
    uint64_t hi_unmixed = lo_unmixed + GOLDEN_RATIO;
    uint64_t s0 = mix_stafford13(lo_unmixed);
    uint64_t s1 = mix_stafford13(hi_unmixed);

    if ((s0 | s1) == 0) {
        s0 = GOLDEN_RATIO;
        s1 = SILVER_RATIO;
    }

    uint64_t a = worldgen_next_long(s0, s1) | 1ULL;
    uint64_t b = worldgen_next_long(s0, s1) | 1ULL;

    uint64_t sum_low = (a + b) & MASK60;
    uint64_t diff_low = (a - b) & MASK60;

    if (sum_low == 0 || diff_low == 0) {
        uint32_t slot = atomicAdd(result_count, 1);
        if (slot < max_results) {
            results[slot * 3 + 0] = world_seed;
            results[slot * 3 + 1] = a;
            results[slot * 3 + 2] = b;
        }
    }
}

void print_result(uint64_t seed, uint64_t a, uint64_t b) {
    uint64_t diff_low = (a - b) & MASK60;

    const char *dir;
    if (diff_low == 0) dir = "(-1,1)";
    else dir = "(1,1)";

    int64_t signed_seed = (int64_t)seed;
    printf("FOUND %" PRIu64 " %" PRId64 " %s a=0x%016" PRIX64 " b=0x%016" PRIX64 "\n",
           seed, signed_seed, dir, a, b);
    fflush(stdout);
}

int main(int argc, char **argv) {
    uint64_t seed_start = 0;
    uint64_t seed_count = (1ULL << 32);

    if (argc >= 3) {
        seed_start = strtoull(argv[1], NULL, 0);
        seed_count = strtoull(argv[2], NULL, 0);
    } else if (argc == 2) {
        seed_count = strtoull(argv[1], NULL, 0);
    }

    printf("=== Diagonal Seed Finder for MC 26.1 (CUDA) ===\n");
    printf("Search range: [%" PRIu64 ", %" PRIu64 ")\n", seed_start, seed_start + seed_count);
    printf("Total seeds to check: %" PRIu64 " (2^%.1f)\n",
           seed_count, log2((double)seed_count));

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    const uint32_t MAX_RESULTS = 4096;
    const int THREADS_PER_BLOCK = 256;

    uint64_t batch_size = (uint64_t)THREADS_PER_BLOCK * prop.multiProcessorCount * 512;
    batch_size = ((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;

    printf("Batch size: %" PRIu64 "\n\n", batch_size);

    uint64_t *d_results;
    uint32_t *d_count;
    cudaMalloc(&d_results, MAX_RESULTS * 3 * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(uint32_t));
    uint64_t *h_results = (uint64_t *)malloc(MAX_RESULTS * 3 * sizeof(uint64_t));
    uint32_t h_count;

    uint64_t total_found = 0;
    uint64_t seeds_done = 0;

    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventRecord(start_event);

    float last_report_time = 0;

    for (uint64_t offset = 0; offset < seed_count; offset += batch_size) {
        uint64_t this_batch = batch_size;
        if (offset + this_batch > seed_count) {
            this_batch = seed_count - offset;
        }

        cudaMemset(d_count, 0, sizeof(uint32_t));

        int blocks = (int)((this_batch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        search_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            seed_start + offset, this_batch, d_results, d_count, MAX_RESULTS);

        cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (h_count > 0) {
            uint32_t to_copy = h_count < MAX_RESULTS ? h_count : MAX_RESULTS;
            cudaMemcpy(h_results, d_results, to_copy * 3 * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < to_copy; i++) {
                print_result(h_results[i*3], h_results[i*3+1], h_results[i*3+2]);
            }
            total_found += h_count;
        }

        seeds_done += this_batch;

        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event, end_event);

        if (elapsed_ms - last_report_time > 2000.0f || seeds_done >= seed_count) {
            last_report_time = elapsed_ms;
            double elapsed_sec = elapsed_ms / 1000.0;
            double rate = seeds_done / elapsed_sec;
            double pct = 100.0 * seeds_done / seed_count;
            double eta_sec = (seed_count - seeds_done) / rate;

            fprintf(stderr, "\r[%.1f%%] %" PRIu64 "/%" PRIu64 " seeds | %.2e seeds/s | "
                    "found: %" PRIu64 " | elapsed: %.0fs | ETA: %.0fs   ",
                    pct, seeds_done, seed_count, rate, total_found, elapsed_sec, eta_sec);
        }
    }

    fprintf(stderr, "\n");
    printf("\n=== Done. Found %" PRIu64 " diagonal seed(s). ===\n", total_found);
    cudaFree(d_results);
    cudaFree(d_count);
    free(h_results);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);

    return 0;
}
