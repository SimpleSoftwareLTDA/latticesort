//! Comprehensive integration and property-style test suite for `latticesort`.
//!
//! Validates all algorithmic paths, cutoff boundaries (`384`, `2048`, `200_000`, `1_500_000`),
//! adversarial distributions, polymorphic types (`i32`, `u32`, `i64`, `u64`, `f32`, `f64`),
//! and multi-core execution against `std::slice::sort_unstable` / `sort_by`.

use latticesort::{LatticeSortExt, lattice_sort_i32, lattice_sort_i64, parallel_lattice_sort};

/// Deterministic SplitMix64 pseudo-random generator for reproducible tests.
#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_i32(&mut self) -> i32 {
        self.next_u64() as i32
    }

    fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }
}

fn assert_sorts_i32(mut data: Vec<i32>) {
    let mut expected = data.clone();
    expected.sort_unstable();
    lattice_sort_i32(&mut data);
    assert_eq!(data, expected);
}

#[test]
fn test_degenerate_and_tiny_slices() {
    assert_sorts_i32(vec![]);
    assert_sorts_i32(vec![0]);
    assert_sorts_i32(vec![i32::MIN]);
    assert_sorts_i32(vec![i32::MAX]);
    assert_sorts_i32(vec![1, 2]);
    assert_sorts_i32(vec![2, 1]);
    assert_sorts_i32(vec![5, 5]);
    assert_sorts_i32(vec![i32::MAX, i32::MIN, 0, -1, 1]);

    let mut rng = SplitMix64::new(0x1234_5678);
    for len in 0..=128 {
        let data: Vec<i32> = (0..len).map(|_| rng.next_i32()).collect();
        assert_sorts_i32(data);
    }
}

#[test]
fn test_cutoff_boundaries_384_and_2048() {
    let mut rng = SplitMix64::new(0xCAFE_BABE);
    // Test around SMALL_SORT_CUTOFF (384) and stack scratch boundary (2048)
    let boundary_sizes = [
        380, 383, 384, 385, 386, 400, 511, 512, 513, 1023, 1024, 1025, 2046, 2047, 2048, 2049,
        2050, 4096,
    ];
    for &len in &boundary_sizes {
        let uniform: Vec<i32> = (0..len).map(|_| rng.next_i32()).collect();
        assert_sorts_i32(uniform);

        let low_card: Vec<i32> = (0..len).map(|_| (rng.next_u64() % 17) as i32 - 8).collect();
        assert_sorts_i32(low_card);

        let dense_span: Vec<i32> = (0..len)
            .map(|_| 10_000 + (rng.next_u64() % (len as u64 / 2 + 1)) as i32)
            .collect();
        assert_sorts_i32(dense_span);
    }
}

#[test]
fn test_structural_soliton_and_monotonic_paths() {
    for &len in &[100, 384, 385, 2048, 2049, 10_000] {
        // Strictly ascending
        assert_sorts_i32((0..len as i32).collect());
        // Strictly descending
        assert_sorts_i32((0..len as i32).rev().collect());
        // All equal
        assert_sorts_i32(vec![42; len]);

        // Peak soliton (ascending then descending)
        let mid = len / 2;
        let peak: Vec<i32> = (0..mid as i32)
            .chain((0..(len - mid) as i32).rev())
            .collect();
        assert_sorts_i32(peak);

        // Valley soliton (descending then ascending)
        let valley: Vec<i32> = (0..mid as i32).rev().chain(0..(len - mid) as i32).collect();
        assert_sorts_i32(valley);

        // Fake soliton (90% ascending ramp + random noise tail)
        let mut rng = SplitMix64::new(len as u64);
        let ramp_len = (len * 9) / 10;
        let fake_soliton: Vec<i32> = (0..ramp_len as i32)
            .chain((0..(len - ramp_len)).map(|_| rng.next_i32()))
            .collect();
        assert_sorts_i32(fake_soliton);
    }
}

#[test]
fn test_adversarial_entropy_and_hash_collision_traps() {
    let mut rng = SplitMix64::new(0xDEAD_BEEF);
    let len = 25_000;

    // 1. Iceberg: 95% constant + 5% random tail
    let prefix = (len * 95) / 100;
    let iceberg: Vec<i32> = std::iter::repeat_n(777, prefix)
        .chain((0..(len - prefix)).map(|_| rng.next_i32()))
        .collect();
    assert_sorts_i32(iceberg);

    // 2. Mid-array entropy trap: first 50% only 4 distinct values, second 50% uniform random
    let half = len / 2;
    let mid_trap: Vec<i32> = (0..half)
        .map(|i| (i % 4) as i32)
        .chain((0..(len - half)).map(|_| rng.next_i32()))
        .collect();
    assert_sorts_i32(mid_trap);

    // 3. Multiples of golden ratio hash multiplier (collision stress)
    let collision_stress: Vec<i32> = (0..len)
        .map(|i| ((i % 31) as u32).wrapping_mul(0x9E37_79B9) as i32)
        .collect();
    assert_sorts_i32(collision_stress);
}

#[test]
fn test_large_msb_hybrid_and_parallel_cutoffs() {
    let mut rng = SplitMix64::new(0xFACE_FEED);

    // Test across parallel_lattice_sort threshold (200_000)
    for &len in &[199_999, 200_000, 200_001] {
        let mut data: Vec<i32> = (0..len).map(|_| rng.next_i32()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        parallel_lattice_sort(&mut data);
        assert_eq!(data, expected);
    }

    // Test across MSB_HYBRID_THRESHOLD (1_500_000)
    let mut large: Vec<i32> = (0..1_500_005).map(|_| rng.next_i32()).collect();
    let mut expected_large = large.clone();
    expected_large.sort_unstable();
    lattice_sort_i32(&mut large);
    assert_eq!(large, expected_large);
}

#[test]
fn test_polymorphic_types_and_ieee754_edge_cases() {
    let mut rng = SplitMix64::new(0x7777_8888);

    // u32 including 0 and u32::MAX
    let mut u32_data: Vec<u32> = (0..5_000)
        .map(|_| rng.next_u64() as u32)
        .chain([0, u32::MAX, i32::MAX as u32, (i32::MAX as u32) + 1])
        .collect();
    let mut u32_expected = u32_data.clone();
    u32_expected.sort_unstable();
    u32_data.lattice_sort();
    assert_eq!(u32_data, u32_expected);

    // i64 including i64::MIN, i64::MAX, and timestamp-like shared upper bytes
    let mut i64_data: Vec<i64> = (0..5_000)
        .map(|_| rng.next_i64())
        .chain([i64::MIN, i64::MAX, -1, 0, 1])
        .collect();
    let mut i64_expected = i64_data.clone();
    i64_expected.sort_unstable();
    lattice_sort_i64(&mut i64_data);
    assert_eq!(i64_data, i64_expected);

    // u64 including 0 and u64::MAX
    let mut u64_data: Vec<u64> = (0..5_000)
        .map(|_| rng.next_u64())
        .chain([0, u64::MAX, i64::MAX as u64, (i64::MAX as u64) + 1])
        .collect();
    let mut u64_expected = u64_data.clone();
    u64_expected.sort_unstable();
    u64_data.lattice_sort();
    assert_eq!(u64_data, u64_expected);

    // f32 including -0.0, +0.0, subnormals, infinities, and NaNs
    let mut f32_data: Vec<f32> = (0..2_000)
        .map(|_| (rng.next_i32() as f32) * 0.001)
        .chain([
            0.0_f32,
            -0.0_f32,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ])
        .collect();
    let mut f32_expected = f32_data.clone();
    f32_expected.sort_by(f32::total_cmp);
    f32_data.lattice_sort();
    for (a, b) in f32_data.iter().zip(f32_expected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // f64 including -0.0, +0.0, subnormals, infinities, and NaNs
    let mut f64_data: Vec<f64> = (0..2_000)
        .map(|_| (rng.next_i64() as f64) * 1e-9)
        .chain([
            0.0_f64,
            -0.0_f64,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ])
        .collect();
    let mut f64_expected = f64_data.clone();
    f64_expected.sort_by(f64::total_cmp);
    f64_data.lattice_sort();
    for (a, b) in f64_data.iter().zip(f64_expected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // 64-bit bitonic soliton wave + dense range count + par_lattice_sort
    let mut i64_soliton: Vec<i64> = (0..5_000_i64).chain((0..5_000_i64).rev()).collect();
    let mut i64_soliton_expected = i64_soliton.clone();
    i64_soliton_expected.sort_unstable();
    i64_soliton.lattice_sort();
    assert_eq!(i64_soliton, i64_soliton_expected);

    let mut u64_par: Vec<u64> = (0..220_000).map(|_| rng.next_u64()).collect();
    let mut u64_par_expected = u64_par.clone();
    u64_par_expected.sort_unstable();
    u64_par.par_lattice_sort();
    assert_eq!(u64_par, u64_par_expected);
}

#[test]
fn test_degenerate_byte_combinations_32_and_64() {
    let mut rng = SplitMix64::new(0xABCD_EF01);
    let len = 4096_usize;

    // 32-bit: 2 active bytes (Byte 0 + Byte 3 active, Bytes 1 & 2 constant -> > 384 keys, span > 1M)
    let mut two_bytes_32: Vec<i32> = (0..len)
        .map(|_| {
            let r = rng.next_u64() as u32;
            ((r & 0x0000_00ff) | (r & 0xff00_0000) | 0x00AA_5500) as i32
        })
        .collect();
    let mut two_bytes_32_expected = two_bytes_32.clone();
    two_bytes_32_expected.sort_unstable();
    lattice_sort_i32(&mut two_bytes_32);
    assert_eq!(two_bytes_32, two_bytes_32_expected);

    // 32-bit: 3 active bytes (Bytes 0, 2, 3 active -> odd number of radix passes -> in_scratch copy-back)
    let mut three_bytes_32: Vec<i32> = (0..len)
        .map(|_| {
            let r = rng.next_u64() as u32;
            ((r & 0xffff_00ff) | 0x0000_5A00) as i32
        })
        .collect();
    let mut three_bytes_32_expected = three_bytes_32.clone();
    three_bytes_32_expected.sort_unstable();
    lattice_sort_i32(&mut three_bytes_32);
    assert_eq!(three_bytes_32, three_bytes_32_expected);

    // 64-bit: 3 active bytes (Bytes 0, 4, 7 active -> odd number of passes via public API)
    let mut three_bytes_64: Vec<i64> = (0..len)
        .map(|_| {
            let r = rng.next_u64();
            let active = (r & 0xff) | (r & 0x0000_00ff_0000_0000) | (r & 0xff00_0000_0000_0000);
            (active | 0x0055_5555_0055_5500) as i64
        })
        .collect();
    let mut three_bytes_64_expected = three_bytes_64.clone();
    three_bytes_64_expected.sort_unstable();
    lattice_sort_i64(&mut three_bytes_64);
    assert_eq!(three_bytes_64, three_bytes_64_expected);
}

#[test]
fn test_parallel_degenerate_msb_and_ieee754_total_order() {
    let mut rng = SplitMix64::new(0x5555_AAAA);
    let len = 210_000_usize;

    // All elements share the exact same MSB (Byte 3 == 0x2A), triggering msb_histogram.contains(&len).
    let mut i32_same_msb: Vec<i32> = (0..len)
        .map(|_| ((0x2A00_0000_u32) | ((rng.next_u64() as u32) & 0x00ff_ffff)) as i32)
        .collect();
    let mut i32_same_msb_expected = i32_same_msb.clone();
    i32_same_msb_expected.sort_unstable();
    parallel_lattice_sort(&mut i32_same_msb);
    assert_eq!(i32_same_msb, i32_same_msb_expected);

    // 64-bit parallel with identical MSB (Byte 7 == 0x42).
    let mut i64_same_msb: Vec<i64> = (0..len)
        .map(|_| ((0x4200_0000_0000_0000_u64) | (rng.next_u64() & 0x00ff_ffff_ffff_ffff)) as i64)
        .collect();
    let mut i64_same_msb_expected = i64_same_msb.clone();
    i64_same_msb_expected.sort_unstable();
    i64_same_msb.par_lattice_sort();
    assert_eq!(i64_same_msb, i64_same_msb_expected);

    // Parallel IEEE-754 total order for f32 and f64 with signed zeros, subnormals, infinities, NaNs.
    let mut f32_par: Vec<f32> = (0..len)
        .map(|i| match i % 256 {
            0 => 0.0_f32,
            1 => -0.0_f32,
            2 => f32::MIN_POSITIVE,
            3 => -f32::MIN_POSITIVE,
            4 => f32::INFINITY,
            5 => f32::NEG_INFINITY,
            6 => f32::NAN,
            7 => -f32::NAN,
            _ => (rng.next_i32() as f32) * 0.01,
        })
        .collect();
    let mut f32_par_expected = f32_par.clone();
    f32_par_expected.sort_by(f32::total_cmp);
    f32_par.par_lattice_sort();
    for (a, b) in f32_par.iter().zip(f32_par_expected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let mut f64_par: Vec<f64> = (0..len)
        .map(|i| match i % 256 {
            0 => 0.0_f64,
            1 => -0.0_f64,
            2 => f64::MIN_POSITIVE,
            3 => -f64::MIN_POSITIVE,
            4 => f64::INFINITY,
            5 => f64::NEG_INFINITY,
            6 => f64::NAN,
            7 => -f64::NAN,
            _ => (rng.next_i64() as f64) * 1e-6,
        })
        .collect();
    let mut f64_par_expected = f64_par.clone();
    f64_par_expected.sort_by(f64::total_cmp);
    f64_par.par_lattice_sort();
    for (a, b) in f64_par.iter().zip(f64_par_expected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
