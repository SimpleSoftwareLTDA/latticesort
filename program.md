# Autoresearch Program: Natural & Sacred Geometry Integer Sorting in Rust

## Objective
Discover and validate integer sorting algorithms in Rust inspired by sacred geometry, mathematical constants of nature ($\phi, \pi, e, \delta, \alpha$), and physical phenomena (crystal nucleation, barycentric condensation, wave solitons, atomic spectral shells).

The goal metric is **Composite Geometric Mean Latency (`geomean_ms`)** across 9 canonical distributions at $N = 1,000,000$ (lower is better), while strictly maintaining:
1. **100% Exactness**: Every output slice must match `std::slice::sort_unstable` element-for-element.
2. **Systems Safety**: Zero `unsafe` blocks (or strict `// SAFETY:` contracts), zero warnings in `cargo clippy --all-targets -- -D warnings`.
3. **Simplicity Bias**: Keep only mechanisms that measurably improve `geomean_ms` or win a distribution class without degrading the geometric mean.

## Three-File Autoresearch Architecture
1. `src/harness.rs` (**LOCKED / READ-ONLY**): Generates the 9 deterministic benchmark distributions (including tight duplicates, wide-spread duplicates, primes, Pareto/Zipf power laws, natural wave oscillations, and full 32-bit uniform random), verifies correctness against `std::slice::sort_unstable`, and computes `geomean_ms`.
2. `src/candidate.rs` (**MODIFIABLE TARGET**): Contains the active candidate sorting algorithm (`candidate_sort`) under investigation.
3. `results.tsv` & `RESEARCH_LOG.md`: Append-only record of every experiment, hypothesis, natural/geometric inspiration, `geomean_ms`, speedup vs `std::slice::sort_unstable`, and `KEEP` / `DISCARD` decision backed by Git commits.
