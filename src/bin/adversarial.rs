#[path = "../candidate.rs"]
mod candidate;

use std::time::Instant;

fn measure_median_us(base: &[i32], buf: &mut [i32], sort_fn: fn(&mut [i32]), trials: usize) -> f64 {
    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        buf.copy_from_slice(base);
        let t0 = Instant::now();
        sort_fn(buf);
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[trials / 2]
}

/// 1A. Iceberg Attack (95% constant 0 + 5% random 32-bit tail).
fn make_iceberg_attack(n: usize) -> Vec<i32> {
    let mut out = vec![0_i32; n];
    let tail_start = (n * 95) / 100;
    let mut seed: u32 = 0xDEAD_BEEF;
    for slot in &mut out[tail_start..] {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *slot = seed as i32;
    }
    out
}

/// 1B. Mid-Array Entropy Trap: first 45% has 15 keys, middle 20% has 150 keys (triggers abort
/// at ~55% of the scan after wasting 500,000 hash lookups!), last 35% has 15 keys.
fn make_mid_entropy_trap(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 0xCAFE_BABE;
    for i in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let k = if i > (n * 45) / 100 && i < (n * 65) / 100 {
            (seed % 200) as i32
        } else {
            (seed % 10) as i32
        };
        out.push((k - 100).wrapping_mul(15_000_000));
    }
    out
}

/// 1C. Sentinel Value & Hash Collision Torture (0, 1, -1, i32::MIN, i32::MAX + 19 colliding keys).
fn make_sentinel_torture(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let target_slot = {
        let u = 0_u32;
        (((u ^ (u >> 15)).wrapping_mul(0x9E37_79B9)) >> 23) as usize
    };
    let mut colliding_keys = vec![0_i32, 1_i32, -1_i32, i32::MIN, i32::MAX];
    let mut candidate_val = 2_i32;
    while colliding_keys.len() < 24 {
        let u = candidate_val as u32;
        let s = (((u ^ (u >> 15)).wrapping_mul(0x9E37_79B9)) >> 23) as usize;
        if s == target_slot {
            colliding_keys.push(candidate_val);
        }
        candidate_val += 1;
    }

    let mut seed: u32 = 424242;
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

/// 2A. Cutoff Boundary Attack (N = 600, just above the 512 cutoff where 4x256 radix histograms
/// pay fixed 1024-counter overhead on only 600 elements).
fn make_cutoff_boundary(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 314_159_265;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        out.push(seed as i32);
    }
    out
}

/// 2B. Mid-Point Single Swap (`swap(n/2, n/2 + 1)` on an otherwise sorted wide array).
/// Defeats Soliton (peak = 50% < 75%), Nucleation, and Barycentric, while `std::sort_unstable`
/// detects that both partitions around the median are already sorted!
fn make_midpoint_single_swap(n: usize) -> Vec<i32> {
    let mut out: Vec<i32> = (0..n).map(|i| (i as i32 - 500_000).wrapping_mul(1000)).collect();
    out.swap(n / 2 - 2, n / 2 - 1);
    out
}

/// 3A. Fake Soliton (92% ascending ramp, then oscillating tail).
fn make_fake_soliton(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let cutoff = (n * 92) / 100;
    for i in 0..cutoff {
        out.push((i as i32).wrapping_mul(500));
    }
    for i in cutoff..n {
        let sign = if i % 2 == 0 { 1 } else { -1 };
        out.push(((n - i) as i32).wrapping_mul(500) * sign);
    }
    out
}

/// 3B. Valley Soliton (U-shape: descending first half, ascending second half).
fn make_valley_soliton(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..half {
        out.push(((half - i) as i32).wrapping_mul(1000));
    }
    for i in half..n {
        out.push(((i - half) as i32).wrapping_mul(1000));
    }
    out
}

/// 4A. Sparse Barycentric Horizon (K=150 > 128 crystal limit, span = 1,000,000).
fn make_sparse_barycentric_150(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    let mut seed: u32 = 777_777;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let k = (seed % 150) as i32;
        out.push((k * 1_000_000) / 149);
    }
    out
}

/// 4B.Exact Golden-Stride Spoof: plants constant 0 at the exact 96 golden-stride indices AND
/// the 32 tail indices probed by `try_phyllotaxis_nucleation`, while all other elements are random!
fn make_exact_golden_stride_spoof(n: usize) -> Vec<i32> {
    let mut out = make_cutoff_boundary(n);
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
    let n = 1_000_000;
    let attacks: Vec<(&str, Vec<i32>, usize)> = vec![
        ("1A. Iceberg Attack (95% const + 5% rand)", make_iceberg_attack(n), 3),
        ("1B. Mid-Array Entropy Trap (abort @ 50%)", make_mid_entropy_trap(n), 3),
        ("1C. Sentinel & Hash Collision Torture", make_sentinel_torture(n), 3),
        ("2A. Cutoff Boundary (N = 600 uniform)", make_cutoff_boundary(600), 51),
        ("2B. Mid-Point Single Swap (swap near n/2)", make_midpoint_single_swap(n), 3),
        ("3A. Fake Soliton (92% ramp + noise tail)", make_fake_soliton(n), 3),
        ("3B. Valley Soliton (U-shape trough->crest)", make_valley_soliton(n), 3),
        ("4A. Sparse Barycentric (K=150, span 10^6)", make_sparse_barycentric_150(n), 3),
        ("4B. Exact Golden-Stride Probe Spoof", make_exact_golden_stride_spoof(n), 3),
    ];

    println!("=== RED-TEAM FALSIFICATION SUITE (ADVERSARIAL TESTS) ===");
    println!(
        "{:<43} | {:>12} | {:>12} | {:>9} | {:>9}",
        "Adversarial Attack", "Cand (us)", "Std (us)", "Ratio", "Verdict"
    );
    println!("{}", "-".repeat(95));

    for (label, data, trials) in &attacks {
        let mut work_cand = vec![0_i32; data.len()];
        let mut work_std = vec![0_i32; data.len()];

        let std_us = measure_median_us(data, &mut work_std, <[i32]>::sort_unstable, *trials);
        let cand_us = measure_median_us(data, &mut work_cand, candidate::candidate_sort, *trials);

        assert_eq!(
            work_cand, work_std,
            "CORRUPTION DETECTED on adversarial attack: {label}"
        );

        let ratio = std_us / cand_us;
        let verdict = if ratio >= 1.0 { "SURVIVED" } else { "FALSIFIED" };
        println!(
            "{:<43} | {:>12.1} | {:>12.1} | {:>8.2}x | {:>9}",
            label, cand_us, std_us, ratio, verdict
        );
    }
}
