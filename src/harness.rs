//! Evaluation harness and synthetic dataset generators for `latticesort`.

use std::time::Instant;

/// Timing and speedup metrics for a single benchmark dataset.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CaseResult {
    /// Human-readable label of the benchmark dataset.
    pub label: &'static str,
    /// Median execution time of the candidate sorting algorithm in milliseconds.
    pub candidate_ms: f64,
    /// Median execution time of `std::slice::sort_unstable` in milliseconds.
    pub std_ms: f64,
    /// Ratio `std_ms / candidate_ms` (values above `1.0` indicate speedup over `std`).
    pub speedup: f64,
}

/// Aggregated benchmark report across all datasets in the evaluation suite.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SuiteReport {
    /// Per-dataset timing and speedup results.
    pub cases: Vec<CaseResult>,
    /// Geometric mean execution time of the candidate algorithm in milliseconds.
    pub candidate_geomean_ms: f64,
    /// Geometric mean execution time of `std::slice::sort_unstable` in milliseconds.
    pub std_geomean_ms: f64,
    /// Overall geometric mean speedup ratio (`std_geomean_ms / candidate_geomean_ms`).
    pub overall_speedup: f64,
}

struct BenchmarkDataset {
    label: &'static str,
    values: Vec<i32>,
}

/// Generates a deterministic cyclic modulo-$k$ dataset of length `n`.
#[must_use]
pub fn make_mod_k(n: usize, k: i32) -> Vec<i32> {
    (0..n)
        .map(|i| ((i as i64 * 7) % i64::from(k)) as i32)
        .collect()
}

/// Generates a low-cardinality dataset with $k$ distinct values spread across a wide $10^9$ span.
#[must_use]
pub fn make_wide_mod_k(n: usize, k: u32) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 987_654_321;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let bucket = (seed % k) as i32 - (k as i32 / 2);
        out.push(bucket.wrapping_mul(80_000_000));
    }
    out
}

/// Generates pseudo-random values uniformly distributed in `0..k`.
#[must_use]
pub fn make_random_mod_k(n: usize, k: u32) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 133_742;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push((seed % k) as i32);
    }
    out
}

/// Generates a bitonic wave (ascending first half, descending second half).
#[must_use]
pub fn make_soliton_wave(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..half {
        out.push(i as i32 * 3);
    }
    for i in half..n {
        out.push((n - i) as i32 * 3);
    }
    out
}

/// Generates a skewed Pareto power-law distribution with heavy clustering near zero.
#[must_use]
pub fn make_pareto_power_law(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 271_828_182;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let denom = ((seed >> 16) & 0x0fff) + 1;
        let val = (seed & 0x000f_ffff) / denom;
        out.push(val as i32);
    }
    out
}

/// Generates uniformly distributed 32-bit signed integers across the full `i32` range.
#[must_use]
pub fn make_uniform_random(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 314_159_265;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push(seed as i32);
    }
    out
}

/// Verifies `sort_fn` against `std::slice::sort_unstable` across boundary and edge cases.
///
/// # Panics
///
/// Panics if `sort_fn` produces a slice that differs from `std::slice::sort_unstable`.
pub fn verify_edge_cases(sort_fn: fn(&mut [i32])) {
    let cases: Vec<Vec<i32>> = vec![
        vec![],
        vec![0],
        vec![42, -42],
        vec![-1, -1, -1, -1, -1],
        vec![i32::MAX, i32::MIN, 0, -1, 1, i32::MAX, i32::MIN],
        (0..257).rev().collect(),
        make_mod_k(1024, 13),
        make_wide_mod_k(1024, 13),
        make_uniform_random(2048),
    ];

    for case in cases {
        let mut expected = case.clone();
        expected.sort_unstable();
        let mut actual = case;
        sort_fn(&mut actual);
        assert_eq!(actual, expected, "Candidate failed edge-case verification");
    }
}

fn measure_median_ms(base: &[i32], buf: &mut [i32], sort_fn: fn(&mut [i32]), trials: usize) -> f64 {
    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        buf.copy_from_slice(base);
        let t0 = Instant::now();
        sort_fn(buf);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[trials / 2]
}

/// Evaluates `sort_fn` across all 9 benchmark distributions of length `n`.
///
/// # Panics
///
/// Panics if `sort_fn` produces an improperly sorted result on any dataset.
#[must_use]
pub fn evaluate_candidate(sort_fn: fn(&mut [i32]), n: usize) -> SuiteReport {
    verify_edge_cases(sort_fn);

    let datasets: Vec<BenchmarkDataset> = vec![
        BenchmarkDataset {
            label: "1. Cyclic K=12 (non-13)",
            values: make_mod_k(n, 12),
        },
        BenchmarkDataset {
            label: "2. Cyclic K=13 (base-13)",
            values: make_mod_k(n, 13),
        },
        BenchmarkDataset {
            label: "3. Cyclic K=17 (prime)",
            values: make_mod_k(n, 17),
        },
        BenchmarkDataset {
            label: "4. Wide K=13 (span 10^9)",
            values: make_wide_mod_k(n, 13),
        },
        BenchmarkDataset {
            label: "5. Random K=100",
            values: make_random_mod_k(n, 100),
        },
        BenchmarkDataset {
            label: "6. Random K=1000",
            values: make_random_mod_k(n, 1000),
        },
        BenchmarkDataset {
            label: "7. Soliton Wave (bimodal)",
            values: make_soliton_wave(n),
        },
        BenchmarkDataset {
            label: "8. Pareto Power-Law",
            values: make_pareto_power_law(n),
        },
        BenchmarkDataset {
            label: "9. Uniform 32-bit Random",
            values: make_uniform_random(n),
        },
    ];

    let mut cases = Vec::with_capacity(datasets.len());
    let mut log_sum_candidate = 0.0_f64;
    let mut log_sum_std = 0.0_f64;
    let mut work_candidate = vec![0_i32; n];
    let mut work_std = vec![0_i32; n];

    for dataset in &datasets {
        let std_ms = measure_median_ms(&dataset.values, &mut work_std, <[i32]>::sort_unstable, 3);
        let candidate_ms = measure_median_ms(&dataset.values, &mut work_candidate, sort_fn, 3);

        assert_eq!(
            work_candidate, work_std,
            "Candidate produced incorrect ordering on dataset: {}",
            dataset.label
        );

        let speedup = std_ms / candidate_ms;
        log_sum_candidate += candidate_ms.ln();
        log_sum_std += std_ms.ln();

        cases.push(CaseResult {
            label: dataset.label,
            candidate_ms,
            std_ms,
            speedup,
        });
    }

    let count = datasets.len() as f64;
    let candidate_geomean_ms = (log_sum_candidate / count).exp();
    let std_geomean_ms = (log_sum_std / count).exp();
    let overall_speedup = std_geomean_ms / candidate_geomean_ms;

    SuiteReport {
        cases,
        candidate_geomean_ms,
        std_geomean_ms,
        overall_speedup,
    }
}
