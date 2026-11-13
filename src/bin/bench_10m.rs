//! 10x scale benchmark and red-team adversarial suite at $N = 10{,}000{,}000$.

use latticesort::{candidate, harness};
use std::time::Instant;

struct AdversarialAttack {
    label: &'static str,
    values: Vec<i32>,
    trials: usize,
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

fn make_iceberg_attack(n: usize) -> Vec<i32> {
    let mut out = vec![0_i32; n];
    let tail_start = (n * 95) / 100;
    let mut seed: u32 = 0xdead_beef;
    for slot in &mut out[tail_start..] {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *slot = seed as i32;
    }
    out
}

fn make_mid_entropy_trap(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 0xcafe_babe;
    for i in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = if i > (n * 45) / 100 && i < (n * 65) / 100 {
            (seed % 200) as i32
        } else {
            (seed % 10) as i32
        };
        out.push((key - 100).wrapping_mul(15_000_000));
    }
    out
}

fn make_sentinel_torture(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let target_slot = {
        let u = 0_u32;
        (((u ^ (u >> 15)).wrapping_mul(0x9e37_79b9)) >> 23) as usize
    };
    let mut colliding_keys = vec![0_i32, 1_i32, -1_i32, i32::MIN, i32::MAX];
    let mut candidate_val = 2_i32;
    while colliding_keys.len() < 24 {
        let u = candidate_val as u32;
        let slot = (((u ^ (u >> 15)).wrapping_mul(0x9e37_79b9)) >> 23) as usize;
        if slot == target_slot {
            colliding_keys.push(candidate_val);
        }
        candidate_val += 1;
    }

    let mut seed: u32 = 424_242;
    for i in 0..n {
        let step = n / 128;
        if i % step == 0 {
            out.push(colliding_keys[5 + (i / step) % (colliding_keys.len() - 5)]);
        } else {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let idx = (seed as usize) % colliding_keys.len();
            out.push(colliding_keys[idx]);
        }
    }
    out
}

fn make_midpoint_single_swap(n: usize) -> Vec<i32> {
    let mut out: Vec<i32> = (0..n)
        .map(|i| (i as i32 - 5_000_000).wrapping_mul(200))
        .collect();
    out.swap(n / 2 - 2, n / 2 - 1);
    out
}

fn make_fake_soliton(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let cutoff = (n * 92) / 100;
    for i in 0..cutoff {
        out.push((i as i32).wrapping_mul(100));
    }
    for i in cutoff..n {
        let sign = if i % 2 == 0 { 1 } else { -1 };
        out.push(((n - i) as i32).wrapping_mul(100) * sign);
    }
    out
}

fn make_valley_soliton(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..half {
        out.push(((half - i) as i32).wrapping_mul(200));
    }
    for i in half..n {
        out.push(((i - half) as i32).wrapping_mul(200));
    }
    out
}

fn make_sparse_barycentric_150(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 777_777;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = (seed % 150) as i32;
        out.push((key * 1_000_000) / 149);
    }
    out
}

fn make_exact_golden_stride_spoof(n: usize) -> Vec<i32> {
    let mut out = harness::make_uniform_random(n);
    let stride = ((n as u64 * 2_654_435_761) >> 32).max(1) as usize | 1;
    let mut idx = n / 5;
    for _ in 0..96 {
        out[idx % n] = 0;
        idx = idx.wrapping_add(stride);
    }
    for slot in &mut out[n - 32..] {
        *slot = 0;
    }
    out
}

fn main() {
    let n = 10_000_000;
    println!("=== 1. MAIN AUTORESEARCH SUITE AT 10x SCALE (N = {n}) ===");
    let report = harness::evaluate_candidate(candidate::lattice_sort_i32, n);
    println!(
        "{:<28} | {:>12} | {:>12} | {:>10}",
        "Dataset", "Lattice (ms)", "Std (ms)", "Speedup"
    );
    println!("{}", "-".repeat(70));
    for case in &report.cases {
        println!(
            "{:<28} | {:>12.2} | {:>12.2} | {:>9.2}x",
            case.label, case.candidate_ms, case.std_ms, case.speedup
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "{:<28} | {:>12.2} | {:>12.2} | {:>9.2}x",
        "GEOMETRIC MEAN (10M)",
        report.candidate_geomean_ms,
        report.std_geomean_ms,
        report.overall_speedup
    );

    println!("\n=== 2. RED-TEAM ADVERSARIAL SUITE AT 10x SCALE (N = {n}) ===");
    let attacks: Vec<AdversarialAttack> = vec![
        AdversarialAttack {
            label: "1A. Iceberg Attack (95% const + 5% rand)",
            values: make_iceberg_attack(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "1B. Mid-Array Entropy Trap (abort @ 50%)",
            values: make_mid_entropy_trap(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "1C. Sentinel & Hash Collision Torture",
            values: make_sentinel_torture(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "2A. 10x Cutoff Boundary (N = 6,000 uniform)",
            values: harness::make_uniform_random(6_000),
            trials: 21,
        },
        AdversarialAttack {
            label: "2B. Mid-Point Single Swap (swap near n/2)",
            values: make_midpoint_single_swap(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "3A. Fake Soliton (92% ramp + noise tail)",
            values: make_fake_soliton(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "3B. Valley Soliton (U-shape trough->crest)",
            values: make_valley_soliton(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "4A. Sparse Barycentric (K=150, span 10^6)",
            values: make_sparse_barycentric_150(n),
            trials: 3,
        },
        AdversarialAttack {
            label: "4B. Exact Golden-Stride Probe Spoof",
            values: make_exact_golden_stride_spoof(n),
            trials: 3,
        },
    ];

    println!(
        "{:<43} | {:>12} | {:>12} | {:>9} | {:>9}",
        "Adversarial Attack (10M)", "Lattice (ms)", "Std (ms)", "Ratio", "Verdict"
    );
    println!("{}", "-".repeat(95));

    for attack in &attacks {
        let mut work_candidate = vec![0_i32; attack.values.len()];
        let mut work_std = vec![0_i32; attack.values.len()];

        let std_ms = measure_median_ms(
            &attack.values,
            &mut work_std,
            <[i32]>::sort_unstable,
            attack.trials,
        );
        let candidate_ms = measure_median_ms(
            &attack.values,
            &mut work_candidate,
            candidate::lattice_sort_i32,
            attack.trials,
        );

        assert_eq!(
            work_candidate, work_std,
            "Corruption detected on 10M adversarial attack: {}",
            attack.label
        );

        let ratio = std_ms / candidate_ms;
        let verdict = if ratio >= 1.0 {
            "SURVIVED"
        } else {
            "FALSIFIED"
        };
        println!(
            "{:<43} | {:>12.3} | {:>12.3} | {:>8.2}x | {:>9}",
            attack.label, candidate_ms, std_ms, ratio, verdict
        );
    }
}
