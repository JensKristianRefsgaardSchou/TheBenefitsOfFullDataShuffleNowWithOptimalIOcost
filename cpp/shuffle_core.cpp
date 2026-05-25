#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>

// ── RNG (one per thread/process) ────────────────────────────────────────────
static thread_local std::mt19937_64 rng{std::random_device{}()};

static inline long long pos_mod(long long a, long long n) {
    return ((a % n) + n) % n;
}

// ── Fisher-Yates on a raw int array ─────────────────────────────────────────
static void fisher_yates(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(arr[i], arr[j]);
    }
}

// ── Fisher-Yates on a vector of vectors (shuffle blocks) ────────────────────
static void shuffle_blocks(std::vector<std::vector<int>>& blocks) {
    int n = (int)blocks.size();
    for (int i = n - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(blocks[i], blocks[j]);
    }
}

// ── Shotgun Shuffle (Extended) ──────────────────────────────────────────────
// Matches shotgun_shuffle_ext from shotgunShuffle.py
static void shotgun_shuffle_ext(int* data, int N, int B, int rounds) {
    if (N == 0) return;
    std::vector<int> buf(N);

    for (int r = 0; r < rounds; r++) {
        int num_blocks = (N + B - 1) / B;

        // Split into blocks, shuffle each internally
        std::vector<std::vector<int>> blocks(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
            int start = i * B;
            int end   = std::min(start + B, N);
            blocks[i].assign(data + start, data + end);
            fisher_yates(blocks[i].data(), (int)blocks[i].size());
        }

        // Shuffle the block order
        shuffle_blocks(blocks);

        // Transpose within gates of B blocks
        int idx = 0;
        for (int g = 0; g < num_blocks; g += B) {
            int gate_end = std::min(g + B, num_blocks);
            // Max block length in this gate
            int max_len = 0;
            for (int b = g; b < gate_end; b++)
                max_len = std::max(max_len, (int)blocks[b].size());
            // Transpose: position-within-block first, then block index
            for (int i = 0; i < max_len; i++) {
                for (int b = g; b < gate_end; b++) {
                    if (i < (int)blocks[b].size())
                        buf[idx++] = blocks[b][i];
                }
            }
        }
        std::memcpy(data, buf.data(), N * sizeof(int));
    }
}

// ── CorgiPile (JK version from main experiment file) ────────────────────────
// 1. Split into blocks of B, shuffle block order
// 2. Group B blocks, flatten each group, fisher-yates the flat group
static void corgi_pile_jk(int* data, int N, int B) {
    if (N == 0) return;
    int num_blocks = (N + B - 1) / B;

    // Split into blocks
    std::vector<std::vector<int>> blocks(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        int start = i * B;
        int end   = std::min(start + B, N);
        blocks[i].assign(data + start, data + end);
    }

    // Shuffle block order
    shuffle_blocks(blocks);

    // Group B blocks, flatten + shuffle each group
    int idx = 0;
    for (int g = 0; g < num_blocks; g += B) {
        int gate_end = std::min(g + B, num_blocks);
        // Flatten
        std::vector<int> flat;
        for (int b = g; b < gate_end; b++)
            flat.insert(flat.end(), blocks[b].begin(), blocks[b].end());
        // Shuffle the flat group
        fisher_yates(flat.data(), (int)flat.size());
        // Write out
        for (int v : flat)
            data[idx++] = v;
    }
}

// ── Modular inverse via Extended GCD ────────────────────────────────────────
static long long extended_gcd(long long a, long long b,
                              long long& x, long long& y) {
    if (a == 0) { x = 0; y = 1; return b; }
    long long x1, y1;
    long long g = extended_gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return g;
}

static long long mod_inverse(long long a, long long n) {
    long long x, y;
    long long g = extended_gcd(pos_mod(a, n), n, x, y);
    if (g != 1) return -1;
    return pos_mod(x, n);
}

// Gen-2-Wise-Ind-Perm.
static void gen_2_wise_ind_perm(int* data, int N, int B) {
    if (N <= 1) return;

    std::vector<int> O(N, 0);
    long long a, a_inv;

    // Pick a ∈ [1, N-1] that has a modular inverse mod N
    do {
        std::uniform_int_distribution<long long> dist(1, (long long)N - 1);
        a = dist(rng);
        a_inv = mod_inverse(a, N);
    } while (a_inv < 0);

    // Precompute (s, s') pairs for b1, b2 ∈ (-B+1 .. B-1)
    int side = 2 * B - 1;
    int num_pairs = side * side;
    std::vector<long long> s_arr(num_pairs), sp_arr(num_pairs);
    {
        int k = 0;
        for (int b1 = -B + 1; b1 < B; b1++) {
            for (int b2 = -B + 1; b2 < B; b2++) {
                s_arr[k]  = a * (long long)b1 + b2;
                sp_arr[k] = (long long)b1 + a_inv * (long long)b2;
                k++;
            }
        }
    }

    // K tracks which indices have been written
    std::vector<char> K(N, 0);
    int next_zero = 0;

    while (next_zero < N) {
        while (next_zero < N && K[next_zero]) next_zero++;
        if (next_zero >= N) break;

        long long i = next_zero;
        long long i_ainv = pos_mod(i * a_inv, (long long)N);

        // Read from i+s, write to i·a_inv + s'
        for (int k = 0; k < num_pairs; k++) {
            int ri = (int)pos_mod(i + s_arr[k], (long long)N);
            int wi = (int)pos_mod(i_ainv + sp_arr[k], (long long)N);
            O[wi] = data[ri];
        }
        // Mark indices i+s as done
        for (int k = 0; k < num_pairs; k++) {
            int ti = (int)pos_mod(i + s_arr[k], (long long)N);
            K[ti] = 1;
        }
    }

    // Random cyclic rotation
    std::uniform_int_distribution<int> rot_dist(0, N - 1);
    int intercept = rot_dist(rng);
    for (int j = 0; j < N; j++)
        data[j] = O[(int)pos_mod((long long)j + N - intercept, (long long)N)];
}

// ── Pair counting (O(N) via frequency map) ──────────────────────────────────
static long long pairs_in_same_output_block(const int* output, int N, int B) {
    long long count = 0;
    int num_blocks = (N + B - 1) / B;
    std::unordered_map<int, int> freq;
    freq.reserve(B + 1);

    for (int i = 0; i < num_blocks; i++) {
        freq.clear();
        int start = i * B;
        int end   = std::min(start + B, N);
        for (int j = start; j < end; j++)
            freq[output[j] / B]++;
        for (auto& [_, v] : freq)
            count += (long long)v * (v - 1) / 2;
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public C API for Python ctypes
// ═══════════════════════════════════════════════════════════════════════════════
#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {

// func_id:  0 = Gen-2-Wise-Ind-Perm
//           1 = IO Shuffle (1 round)
//           2 = IO Shuffle (2 rounds)
//           3 = CorgiPile (1 round)
//           4 = Fisher-Yates
//
// out_counts must point to an array of at least `reps` long longs.
EXPORT void run_worker(int func_id, int N, int B, int reps, long long* out_counts) {
    for (int r = 0; r < reps; r++) {
        std::vector<int> X(N);
        std::iota(X.begin(), X.end(), 0);

        switch (func_id) {
            case 0: gen_2_wise_ind_perm(X.data(), N, B);       break;
            case 1: shotgun_shuffle_ext(X.data(), N, B, 1);    break;
            case 2: shotgun_shuffle_ext(X.data(), N, B, 2);    break;
            case 3: corgi_pile_jk(X.data(), N, B);             break;
            case 4: fisher_yates(X.data(), N);                 break;
        }

        out_counts[r] = pairs_in_same_output_block(X.data(), N, B);
    }
}

// Run a single shuffle and write the output permutation (for scatter plots).
EXPORT void run_single_shuffle(int func_id, int N, int B, int* out) {
    std::iota(out, out + N, 0);
    switch (func_id) {
        case 0: gen_2_wise_ind_perm(out, N, B);       break;
        case 1: shotgun_shuffle_ext(out, N, B, 1);    break;
        case 2: shotgun_shuffle_ext(out, N, B, 2);    break;
        case 3: corgi_pile_jk(out, N, B);             break;
        case 4: fisher_yates(out, N);                 break;
    }
}

EXPORT int get_num_funcs(void) { return 5; }

}  // extern "C"
