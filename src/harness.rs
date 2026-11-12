use std::time::Instant;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CaseResult {
    pub label: &'static str,
    pub candidate_ms: f64,
    pub std_ms: f64,
    pub speedup: f64,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SuiteReport {
    pub cases: Vec<CaseResult>,
    pub candidate_geomean_ms: f64,
    pub std_geomean_ms: f64,
    pub overall_speedup: f64,
}

#[must_use]
pub fn make_mod_k(n: usize, k: i32) -> Vec<i32> {
    (0..n).map(|i| ((i as i64 * 7) % i64::from(k)) as i32).collect()
}

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

#[must_use]
pub fn make_pareto_power_law(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 271_828_182;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let denom = ((seed >> 16) & 0x0FFF) + 1;
        let val = (seed & 0x000F_FFFF) / denom;
        out.push(val as i32);
    }
    out
}

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

#[must_use]
pub fn evaluate_candidate(sort_fn: fn(&mut [i32]), n: usize) -> SuiteReport {
    verify_edge_cases(sort_fn);

    let datasets: Vec<(&'static str, Vec<i32>)> = vec![
        ("1. Cyclic K=12 (non-13)", make_mod_k(n, 12)),
        ("2. Cyclic K=13 (base-13)", make_mod_k(n, 13)),
        ("3. Cyclic K=17 (prime)", make_mod_k(n, 17)),
        ("4. Wide K=13 (span 10^9)", make_wide_mod_k(n, 13)),
        ("5. Random K=100", make_random_mod_k(n, 100)),
        ("6. Random K=1000", make_random_mod_k(n, 1000)),
        ("7. Soliton Wave (bimodal)", make_soliton_wave(n)),
        ("8. Pareto Power-Law", make_pareto_power_law(n)),
        ("9. Uniform 32-bit Random", make_uniform_random(n)),
    ];

    let mut cases = Vec::with_capacity(datasets.len());
    let mut log_sum_cand = 0.0_f64;
    let mut log_sum_std = 0.0_f64;
    let mut work_cand = vec![0_i32; n];
    let mut work_std = vec![0_i32; n];

    for (label, data) in &datasets {
        let std_ms = measure_median_ms(data, &mut work_std, <[i32]>::sort_unstable, 3);
        let candidate_ms = measure_median_ms(data, &mut work_cand, sort_fn, 3);

        assert_eq!(
            work_cand, work_std,
            "Candidate produced incorrect ordering on dataset: {label}"
        );

        let speedup = std_ms / candidate_ms;
        log_sum_cand += candidate_ms.ln();
        log_sum_std += std_ms.ln();

        cases.push(CaseResult {
            label,
            candidate_ms,
            std_ms,
            speedup,
        });
    }

    let count = datasets.len() as f64;
    let candidate_geomean_ms = (log_sum_cand / count).exp();
    let std_geomean_ms = (log_sum_std / count).exp();
    let overall_speedup = std_geomean_ms / candidate_geomean_ms;

    SuiteReport {
        cases,
        candidate_geomean_ms,
        std_geomean_ms,
        overall_speedup,
    }
}
