# Autoresearch Program: `latticesort` Library & Benchmark Architecture

## Objective
Design, verify, and benchmark an adaptive integer and IEEE (Institute of Electrical and Electronics Engineers, international technical standards organization) 754 floating-point sorting library in safe Rust inspired by structural symmetry, dual-sentinel hash lattices, and cache-resident radix cascades.

The primary evaluation metric is **Composite Geometric Mean Latency (`geomean_ms`)** across 9 canonical distributions at $N = 1,000,000$ and $N = 10,000,000$ (lower is better), alongside multi-core throughput up to $N = 1,000,000,000$, while maintaining:
1. **100% Exactness**: Every output slice must match `std::slice::sort_unstable` (or `f32::total_cmp` / `f64::total_cmp`) element-for-element across unit tests, property boundary tests, and benchmark suites.
2. **Systems Safety & Style**: Zero `unsafe` blocks, adherence to the Google Rust Style Guide, and zero warnings under `cargo clippy --all-targets -- -D warnings`.
3. **Adversarial Resilience**: Every candidate optimization must pass the 9-attack Red-Team falsification suite without worst-case degradation.

## Public API
Exposed by `src/lib.rs`:
* `lattice_sort_i32(&mut [i32])`: Single-core adaptive sort for 32-bit signed integers.
* `lattice_sort_i64(&mut [i64])`: Single-core adaptive 8-pass radix and structural sort for 64-bit signed integers.
* `parallel_lattice_sort(&mut [i32])`: Multi-core scoped parallel MSB (Most Significant Byte, highest-order byte of an integer word) + L2 (Level 2 Cache, processor secondary data cache) hybrid radix sort for large `i32` slices.
* `LatticeSortExt`: Extension trait providing `.lattice_sort()` on `[i32]`, `[u32]`, `[i64]`, `[u64]`, `[f32]`, and `[f64]`.

## Repository Layout
1. `src/lib.rs`: Crate root, module exports, and public documentation.
2. `src/candidate.rs`: Core 32-bit adaptive pipeline (`lattice_sort_i32`), including prefix peeling, peak/valley soliton wave merging, 1024-slot dual-sentinel hash aggregation, dense range count sort, fused 4-pass LSD (Least Significant Digit, lowest-order byte pass) radix sort, and 2-level MSB+L2 hybrid cascade.
3. `src/kernel_i64.rs`: 64-bit adaptive 8-pass radix kernel (`lattice_sort_i64`) with constant-byte pass skipping.
4. `src/slice_ext.rs`: Polymorphic trait `LatticeSortExt` with branchless sign-bit and IEEE 754 bijections.
5. `src/parallel.rs`: Multi-core parallel sorting engine (`parallel_lattice_sort`) built on `std::thread::scope`.
6. `src/harness.rs`: Deterministic 9-distribution benchmark evaluator (`N = 1,000,000`).
7. `tests/correctness.rs`: Automated integration and boundary test suite executed by `cargo test`.
8. `src/bin/`:
   * `src/main.rs` (`latticesort`): Primary 1M-element 9-distribution benchmark.
   * `src/bin/adversarial.rs`: Red-Team 9-attack adversarial benchmark at $N = 1,000,000$.
   * `src/bin/bench_10m.rs`: 10M-element main and adversarial benchmark suite.
   * `src/bin/bench_fronts_3_4.rs`: Polymorphic types (`u32`, `i64`, `u64`, `f32`, `f64`) and multi-core benchmark.
   * `src/bin/bench_literature.rs`: Head-to-head comparison against 6 sorting algorithms from Computer Science literature.
   * `src/bin/bench_1b.rs`: 1-Billion-element (`4.00 GB`) single-core and multi-core benchmark.
9. `results.tsv` & `RESEARCH_LOG.md`: Complete chronological record of experiments `exp00` through `exp19`.
