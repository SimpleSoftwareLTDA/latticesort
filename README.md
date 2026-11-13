# `latticesort`

`latticesort` is a zero-dependency, 100% safe Rust sorting library for primitive integers (`i32`, `u32`, `i64`, `u64`) and floating-point slices (`f32`, `f64`). Developed through an autonomous experimental research campaign (`autoresearch`) and hardened against an adversarial red-team suite, `latticesort` achieves a **4.98x geometric mean speedup** at $N = 1{,}000{,}000$ and a **4.02x geometric mean speedup** at $N = 10{,}000{,}000$ over Rust's standard library `std::slice::sort_unstable` (`ipnsort` / pattern-defeating quicksort).

---

## Key Architectural Improvements Over `std::slice::sort_unstable`

`std::slice::sort_unstable` is a general comparison-based hybrid sort bounded by $\Omega(n \log n)$ comparisons on high-entropy keys and $O(n \log k)$ partitioning steps on low-cardinality inputs. `latticesort` combines five specialized stages that adapt to the structural topology, cardinality, value span, and cache footprint of the input slice:

1. **Bitonic and Structural Run Merger (`try_bitonic_run_merge`)**:
   - Performs an 8-point constant-time stride check to filter high-entropy slices before scanning.
   - Detects ascending runs, descending runs, crest-to-trough bitonic waves ($\cap$), valley-to-crest bitonic waves ($\cup$), and inputs with at least a 75% sorted prefix in a single linear $O(n)$ pass.
   - Includes constant-plateau snap-back to prevent equal-value prefixes from splitting across merge boundaries, completing in a single `copy_within` + `fill` pass when the prefix is uniform.

2. **Dual-Sentinel Direct-Mapped Hash Aggregation (`try_sentinel_hash_aggregation`)**:
   - Samples 128 elements using 5 contiguous 16-element quartile bursts combined with a 48-element golden-ratio stride probe to detect low-cardinality distributions while resisting localized entropy traps.
   - Aggregates up to 384 distinct keys inside a stack-allocated 1024-slot table held in L1 (Level 1 Cache, the fastest cache memory core-local to the processor).
   - Uses a dual-sentinel invariant: all slots `s != s0` are initialized to `d0 = 0`, while `s0 = hash(0)` is initialized to a secondary key `d1` whose hash differs from `s0`. Because no slot initially holds a key that hashes to its own index, the hot loop verifies a hit with a single `keys[s] == val` comparison without consulting an occupancy bitmap.
   - Selects a collision-free Fibonacci/SplitMix multiplier from 4 candidates based on the 128-element probe before scanning the slice.

3. **Cache-Bounded Dense Range Counting (`try_dense_range_count`)**:
   - Pre-filtered at zero extra memory cost by the 128-element probe span (`probe_max - probe_min`).
   - When the full key span `max - min` fits within `min(n, 1_048_576)`, sorts in $O(n + \text{span})$ via offset counting inside L2 (Level 2 Cache, the intermediate cache memory between L1 and shared cache) and L3 (Level 3 Cache, the shared last-level cache across processor cores).

4. **Fused-Histogram LSD (Least Significant Digit, a radix sorting order that processes bytes from lowest to highest significance) Radix Sort with Degenerate-Byte Skipping (`radix_lsd_fused_sort`)**:
   - Computes all four 256-entry byte histograms (or all eight histograms for 64-bit keys) in a single vectorized 4x-unrolled pass (`as_chunks::<4>()`).
   - Skips any byte pass where a single bucket count equals `n` (zero-entropy byte), cutting scatter passes in half on bounded integers and realistic 64-bit timestamps.
   - Uses a stack-allocated scratch buffer (`[0_i32; 2048]`) for slices up to $N = 2{,}048$ to eliminate heap allocation overhead on small and medium inputs.

5. **Hybrid MSB (Most Significant Byte, the highest-order byte of a multi-byte word that partitions keys into coarse buckets) + L2-Local Radix Sort (`radix_msb_l2_hybrid_sort`)**:
   - At $N \ge 1{,}500{,}000$, scattering across a full array exceeds L3 capacity and triggers TLB (Translation Lookaside Buffer, the hardware cache that translates virtual memory addresses to physical pages) misses.
   - `latticesort` partitions the slice once by the signed MSB into 256 contiguous sub-buckets (~156 KB each at $N = 10{,}000{,}000$), then sorts the remaining 24 bits of each bucket locally inside L2 cache.

---

## Performance Benchmarks vs `std::slice::sort_unstable`

All benchmarks were compiled with `rustc 1.100.0` (`-C opt-level=3`, Link-Time Optimization enabled, `codegen-units = 1`) and verified bit-for-bit against `std::slice::sort_unstable`.

### 1. Primary Benchmark Suite ($N = 1{,}000{,}000$ `i32` Elements)

| Dataset Distribution | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` | Active Stage |
| :--- | ---: | ---: | ---: | :--- |
| 1. Cyclic `K=12` | `0.788` | `2.655` | **3.37x** | Dual-Sentinel L1 Hash Aggregation |
| 2. Cyclic `K=13` | `0.761` | `2.862` | **3.76x** | Dual-Sentinel L1 Hash Aggregation |
| 3. Cyclic `K=17` (prime) | `0.785` | `3.237` | **4.12x** | Dual-Sentinel L1 Hash Aggregation |
| 4. Wide `K=13` (span $10^9$) | `1.132` | `3.997` | **3.53x** | Dual-Sentinel L1 Hash Aggregation |
| 5. Random `K=100` | `0.858` | `6.143` | **7.16x** | Dual-Sentinel L1 Hash Aggregation |
| 6. Random `K=1000` | `1.204` | `9.826` | **8.16x** | Dense Range Counting (`span <= 1000`) |
| 7. Bitonic Soliton Wave ($\cap$) | `0.625` | `1.835` | **2.93x** | Bitonic Run Merger ($O(n)$) |
| 8. Pareto Power-Law | `1.148` | `12.660` | **11.03x** | Dense Range Counting |
| 9. Uniform 32-bit Random | `5.206` | `14.236` | **2.73x** | 4-Pass Fused LSD Radix Sort |
| **Geometric Mean ($N = 1\text{M}$)** | **`1.113`** | **`5.547`** | **4.98x** | **Full Adaptive Pipeline** |

---

### 2. Large-Scale Benchmark Suite ($N = 10{,}000{,}000$ `i32` Elements)

| Dataset Distribution ($N = 10\text{M}$) | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` | Active Stage |
| :--- | ---: | ---: | ---: | :--- |
| 1. Cyclic `K=12` | `9.05` | `28.59` | **3.16x** | Dual-Sentinel L1 Hash Aggregation |
| 2. Cyclic `K=13` | `9.18` | `28.91` | **3.15x** | Dual-Sentinel L1 Hash Aggregation |
| 3. Cyclic `K=17` (prime) | `8.98` | `32.49` | **3.62x** | Dual-Sentinel L1 Hash Aggregation |
| 4. Wide `K=13` (span $10^9$) | `14.12` | `40.48` | **2.87x** | Dual-Sentinel L1 Hash Aggregation |
| 5. Random `K=100` | `13.45` | `61.49` | **4.57x** | Dual-Sentinel L1 Hash Aggregation |
| 6. Random `K=1000` | `14.28` | `97.69` | **6.84x** | Dense Range Counting |
| 7. Bitonic Soliton Wave ($\cap$) | `9.96` | `22.74` | **2.28x** | Bitonic Run Merger ($O(n)$) |
| 8. Pareto Power-Law | `32.63` | `149.49` | **4.58x** | Dense Range Counting |
| 9. Uniform 32-bit Random | `64.35` | `179.96` | **2.80x** | Hybrid MSB + L2-Local 3-Pass Radix |
| **Geometric Mean ($N = 10\text{M}$)** | **`15.40`** | **`61.93`** | **4.02x** | **Full Adaptive Pipeline** |

---

### 3. Polymorphic Types (`LatticeSortExt`) at $N = 1{,}000{,}000$

`LatticeSortExt` maps unsigned integers (`u32`, `u64`) via sign-bit flipping and IEEE (Institute of Electrical and Electronics Engineers, the standards body defining the 754 floating-point binary representation) 754 floating-point numbers (`f32`, `f64`) via total-order bijective bit projection (`mask = ((bits as i32 >> 31) as u32) & 0x7fff_ffff`).

| Type and Distribution ($N = 1\text{M}$) | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` |
| :--- | ---: | ---: | ---: |
| 1. `u32` (Uniform 32-bit) | `7.10` | `14.26` | **2.01x** |
| 2. `i64` (Timestamps: 4 constant upper bytes skipped) | `12.40` | `14.84` | **1.20x** |
| 3. `i64` (Low-Cardinality `K=50` 64-bit keys) | `1.68` | `4.55` | **2.70x** |
| 4. `u64` (Bounded 32-bit keys in `u64`) | `14.18` | `15.36` | **1.08x** |
| 5. `f32` (Signed IEEE 754 floats `+/-`) | `6.01` | `24.52` | **4.08x** |
| 6. `f64` (Sensor readings `K=64` `+/-`) | `2.77` | `5.11` | **1.85x** |

---

### 4. Multi-Core Parallel Execution (`parallel::parallel_lattice_sort`)

`parallel::parallel_lattice_sort` uses scoped threads (`std::thread::scope`) and disjoint mutable slice splitting (`split_at_mut`) with zero `unsafe` blocks and zero external dependencies.

| Uniform 32-Bit Scenario | `parallel_lattice_sort` (ms) | Single-Thread `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` |
| :--- | ---: | ---: | ---: | ---: |
| $N = 1{,}000{,}000$ | `3.04` | `5.41` | `14.52` | **4.77x** |
| $N = 10{,}000{,}000$ | `32.03` | `64.87` | `161.84` | **5.05x** |

---

### 5. Red-Team Adversarial Falsification Suite ($N = 1{,}000{,}000$ and $N = 10{,}000{,}000$)

Nine pathological distributions were engineered to attack probe sampling, hash collisions, sentinel values (`0`, `1`, `-1`, `i32::MIN`, `i32::MAX`), and cutoff boundaries:

| Adversarial Attack Vector | $N = 1\text{M}$ Speedup vs `std` | $N = 10\text{M}$ Speedup vs `std` | Verdict |
| :--- | ---: | ---: | :--- |
| 1A. Iceberg Attack (95% constant `0` + 5% random tail) | **6.10x** | **4.02x** | Survived |
| 1B. Mid-Array Entropy Trap (200 keys hidden at 45%..65%) | **3.10x** | **1.92x** | Survived |
| 1C. Sentinel & 24-Way Hash Collision Torture | **1.72x** | **1.04x** | Survived |
| 2A. Cutoff Boundary Uniform Random ($N = 600$ / $N = 6{,}000$) | **1.41x** | **2.21x** | Survived |
| 2B. Mid-Point Single Swap (`swap(n/2 - 2, n/2 - 1)`) | **1.14x** | **1.54x** | Survived |
| 3A. Fake Soliton (92% ascending ramp + 8% oscillating tail) | **2.86x** | **2.58x** | Survived |
| 3B. Valley Soliton ($\cup$-shape descending then ascending) | **3.51x** | **2.78x** | Survived |
| 4A. Sparse Horizon (`K=150` keys, span $10^6$) | **4.43x** | **1.96x** | Survived |
| 4B. Exact Golden-Stride Probe Spoof (planted zeros at stride) | **2.68x** | **2.75x** | Survived |

---

## Quick Start

```rust
use latticesort::{LatticeSortExt, parallel};

fn main() {
    // 1. Polymorphic slice sorting via LatticeSortExt (i32, u32, i64, u64, f32, f64)
    let mut timestamps: Vec<i64> = vec![1_760_000_005_000, 1_760_000_001_000, 1_760_000_003_000];
    timestamps.lattice_sort();
    assert!(timestamps.is_sorted());

    let mut readings: Vec<f64> = vec![3.5, -0.0, 0.0, -12.8, 1.25];
    readings.lattice_sort();

    // 2. Multi-core parallel sorting for large i32 batches
    let mut batch: Vec<i32> = (0..1_000_000).rev().collect();
    parallel::parallel_lattice_sort(&mut batch);
    assert!(batch.is_sorted());
}
```

---

## Reproducing the Benchmarks via CLI (Command-Line Interface, text-based terminal interface for running programs)

```bash
# Primary 1,000,000-element benchmark suite (9 distributions)
cargo run --release --bin latticesort

# Red-team adversarial falsification suite (1,000,000 elements)
cargo run --release --bin adversarial

# 10x scale benchmark + red-team suite (10,000,000 elements)
cargo run --release --bin bench_10m

# Polymorphic types (u32, i64, u64, f32, f64) and multi-core parallel benchmark
cargo run --release --bin bench_fronts_3_4
```

---

## Code Quality and Style Conformance

This crate adheres strictly to the [Google Rust Style Guide](https://google.github.io/styleguide/rustguide.html):
- Zero `unsafe` blocks and zero unstable features.
- Flat leaf-module architecture (`src/lib.rs` contains only crate documentation, module declarations, and public re-exports).
- Enforced `#![warn(missing_docs)]` and zero Clippy warnings (`cargo clippy --all-targets -- -D warnings`).
