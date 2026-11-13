//! 1-Billion-element ($N = 1{,}000{,}000{,}000$, $4.0\text{ GB}$ slice) benchmark comparing
//! `parallel::parallel_lattice_sort`, single-threaded `candidate::lattice_sort_i32`, and
//! `std::slice::sort_unstable` (`ipnsort`) on uniform 32-bit random integers.

use latticesort::{candidate, parallel};
use std::num::NonZeroUsize;
use std::thread;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SliceDigest {
    wrapping_sum: u64,
    xor_fold: u32,
}

/// Fills `data` in parallel with a deterministic bijection of index `i` so the exact same
/// 1-billion-element uniform 32-bit random sequence is regenerated in-place without a second
/// 4 GB copy in RAM.
fn fill_deterministic_1b_random(data: &mut [i32]) {
    let len = data.len();
    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(16);
    let chunk_size = len.div_ceil(threads);

    thread::scope(|scope| {
        let indexed_chunks = data.chunks_mut(chunk_size).enumerate();
        for (chunk_idx, chunk) in indexed_chunks {
            let base_idx = chunk_idx * chunk_size;
            scope.spawn(move || {
                let indexed_slots = chunk.iter_mut().enumerate();
                for (offset, slot) in indexed_slots {
                    let mut z = ((base_idx + offset) as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    z ^= z >> 31;
                    *slot = z as i32;
                }
            });
        }
    });
}

/// Computes an order-invariant digest (wrapping 64-bit sum and 32-bit XOR fold) in parallel.
fn compute_digest(data: &[i32]) -> SliceDigest {
    let len = data.len();
    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(16);
    let chunk_size = len.div_ceil(threads);
    let num_chunks = data.chunks(chunk_size).len();
    let mut partial_sums = vec![0_u64; num_chunks];
    let mut partial_xors = vec![0_u32; num_chunks];

    thread::scope(|scope| {
        let iter = data
            .chunks(chunk_size)
            .zip(partial_sums.iter_mut())
            .zip(partial_xors.iter_mut());
        for ((chunk, sum_out), xor_out) in iter {
            scope.spawn(move || {
                let mut local_sum = 0_u64;
                let mut local_xor = 0_u32;
                for &x in chunk {
                    let u = x as u32;
                    local_sum = local_sum.wrapping_add(u64::from(u));
                    local_xor ^= u;
                }
                *sum_out = local_sum;
                *xor_out = local_xor;
            });
        }
    });

    let mut total_sum = 0_u64;
    let mut total_xor = 0_u32;
    for &s in &partial_sums {
        total_sum = total_sum.wrapping_add(s);
    }
    for &x in &partial_xors {
        total_xor ^= x;
    }

    SliceDigest {
        wrapping_sum: total_sum,
        xor_fold: total_xor,
    }
}

fn main() {
    let n: usize = 1_000_000_000;
    let bytes_gb = (n * size_of::<i32>()) as f64 / 1_000_000_000.0;
    println!("=== LATTICESORT 1-BILLION ELEMENT BENCHMARK (N = {n}, {bytes_gb:.2} GB) ===");

    let t_alloc = Instant::now();
    let mut data = vec![0_i32; n];
    fill_deterministic_1b_random(&mut data);
    let expected_digest = compute_digest(&data);
    println!(
        "Generated 1,000,000,000 uniform 32-bit random i32s in {:.2} s (digest sum={:#x}, xor={:#x})\n",
        t_alloc.elapsed().as_secs_f64(),
        expected_digest.wrapping_sum,
        expected_digest.xor_fold
    );

    // 1. Multi-Core Parallel LatticeSort (`parallel_lattice_sort`)
    let t_par = Instant::now();
    parallel::parallel_lattice_sort(&mut data);
    let par_secs = t_par.elapsed().as_secs_f64();
    let par_m_per_sec = (n as f64 / 1_000_000.0) / par_secs;
    assert!(
        data.is_sorted(),
        "1B slice must be strictly sorted after parallel_lattice_sort"
    );
    let par_digest = compute_digest(&data);
    assert_eq!(
        par_digest, expected_digest,
        "parallel_lattice_sort must preserve all 1,000,000,000 elements"
    );
    let par_landmarks = [data[0], data[n / 2], data[n - 1]];
    println!(
        "1. parallel_lattice_sort (Multi-Core): {:>7.3} s ({:>6.1} M elems/s) | min={}, mid={}, max={}",
        par_secs, par_m_per_sec, par_landmarks[0], par_landmarks[1], par_landmarks[2]
    );

    // 2. Single-Threaded LatticeSort (`lattice_sort_i32`)
    fill_deterministic_1b_random(&mut data);
    let t_seq = Instant::now();
    candidate::lattice_sort_i32(&mut data);
    let seq_secs = t_seq.elapsed().as_secs_f64();
    let seq_m_per_sec = (n as f64 / 1_000_000.0) / seq_secs;
    assert!(
        data.is_sorted(),
        "1B slice must be strictly sorted after sequential lattice_sort_i32"
    );
    let seq_digest = compute_digest(&data);
    assert_eq!(
        seq_digest, expected_digest,
        "sequential lattice_sort_i32 must preserve all 1,000,000,000 elements"
    );
    let seq_landmarks = [data[0], data[n / 2], data[n - 1]];
    assert_eq!(
        seq_landmarks, par_landmarks,
        "Sequential and parallel boundary/median elements must match"
    );
    println!(
        "2. lattice_sort_i32 (Single-Thread)  : {:>7.3} s ({:>6.1} M elems/s) | min={}, mid={}, max={}",
        seq_secs, seq_m_per_sec, seq_landmarks[0], seq_landmarks[1], seq_landmarks[2]
    );

    // 3. Rust Standard Library (`std::slice::sort_unstable` / `ipnsort`)
    fill_deterministic_1b_random(&mut data);
    let t_std = Instant::now();
    data.sort_unstable();
    let std_secs = t_std.elapsed().as_secs_f64();
    let std_m_per_sec = (n as f64 / 1_000_000.0) / std_secs;
    assert!(
        data.is_sorted(),
        "1B slice must be strictly sorted after std::slice::sort_unstable"
    );
    let std_digest = compute_digest(&data);
    assert_eq!(
        std_digest, expected_digest,
        "std::slice::sort_unstable digest must match expected digest"
    );
    let std_landmarks = [data[0], data[n / 2], data[n - 1]];
    assert_eq!(
        par_landmarks, std_landmarks,
        "parallel_lattice_sort landmarks must match std::slice::sort_unstable"
    );
    println!(
        "3. std::slice::sort_unstable (1-Thr) : {:>7.3} s ({:>6.1} M elems/s) | min={}, mid={}, max={}",
        std_secs, std_m_per_sec, std_landmarks[0], std_landmarks[1], std_landmarks[2]
    );

    println!("{}", "-".repeat(92));
    println!(
        "SPEEDUP AT N = 1,000,000,000 (1 BILLION):\n  - Single-Thread LatticeSort vs std::sort_unstable : {:.2}x faster ({:.3} s vs {:.3} s)\n  - Multi-Core LatticeSort    vs std::sort_unstable : {:.2}x faster ({:.3} s vs {:.3} s)",
        std_secs / seq_secs,
        seq_secs,
        std_secs,
        std_secs / par_secs,
        par_secs,
        std_secs
    );
}
