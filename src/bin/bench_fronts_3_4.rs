use latticesort::{LatticeSort, candidate, par_lattice_sort};
use std::time::Instant;

fn median_ms<T: Clone, F: FnMut(&mut [T])>(base: &[T], mut sort_fn: F, trials: usize) -> (f64, Vec<T>) {
    let mut times = Vec::with_capacity(trials);
    let mut work = base.to_vec();
    for _ in 0..trials {
        work.clone_from_slice(base);
        let t0 = Instant::now();
        sort_fn(&mut work);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    (times[trials / 2], work)
}

fn main() {
    let n = 1_000_000;
    println!("=== ETAPA 3: GENERALIZACAO DE TIPOS (LatticeSort Trait, N = {n}) ===");
    println!(
        "{:<34} | {:>12} | {:>12} | {:>9} | {:>6}",
        "Tipo e Distribuicao", "Lattice (ms)", "Std (ms)", "Speedup", "OK"
    );
    println!("{}", "-".repeat(83));

    // 1. u32 uniform
    {
        let mut seed = 123_456_789_u32;
        let data: Vec<u32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, <[u32]>::sort_unstable, 3);
        let (lat_ms, act) = median_ms(&data, <[u32]>::lattice_sort, 3);
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "1. u32 (Uniform 32-bit)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            act == exp
        );
        assert_eq!(act, exp);
    }

    // 2. i64 Realistic Timestamps (upper 4 bytes constant -> 4 of 8 bytes skipped!)
    {
        let mut seed = 987_654_321_u64;
        let base_ts: i64 = 1_760_000_000_000;
        let data: Vec<i64> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                base_ts + ((seed >> 32) as i64)
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, <[i64]>::sort_unstable, 3);
        let (lat_ms, act) = median_ms(&data, <[i64]>::lattice_sort, 3);
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "2. i64 (Timestamps: 4 bytes skip)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            act == exp
        );
        assert_eq!(act, exp);
    }

    // 3. i64 Low Cardinality (K = 50 distinct 64-bit IDs)
    {
        let mut seed = 192_837_465_u64;
        let data: Vec<i64> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((seed % 50) as i64 - 25).wrapping_mul(100_000_000_000_000)
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, <[i64]>::sort_unstable, 3);
        let (lat_ms, act) = median_ms(&data, <[i64]>::lattice_sort, 3);
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "3. i64 (K=50 IDs 64-bit)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            act == exp
        );
        assert_eq!(act, exp);
    }

    // 4. u64 Bounded IDs
    {
        let mut seed = 555_666_777_u64;
        let data: Vec<u64> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                seed & 0x0000_0000_FFFF_FFFF
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, <[u64]>::sort_unstable, 3);
        let (lat_ms, act) = median_ms(&data, <[u64]>::lattice_sort, 3);
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "4. u64 (IDs 32-bit em u64)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            act == exp
        );
        assert_eq!(act, exp);
    }

    // 5. f32 IEEE-754 floats (positive + negative + zero)
    {
        let mut seed = 314_159_u32;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed as i32) as f32) * 0.000_1
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, |s| s.sort_unstable_by(f32::total_cmp), 3);
        let (lat_ms, act) = median_ms(&data, <[f32]>::lattice_sort, 3);
        let eq = act.iter().zip(exp.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "5. f32 (Ponto Flutuante +/-)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            eq
        );
        assert!(eq);
    }

    // 6. f64 IEEE-754 doubles (low-cardinality sensor readings K=64)
    {
        let mut seed = 271_828_u32;
        let data: Vec<f64> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from((seed % 64) as i32 - 32) * 1.234_567_89
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, |s| s.sort_unstable_by(f64::total_cmp), 3);
        let (lat_ms, act) = median_ms(&data, <[f64]>::lattice_sort, 3);
        let eq = act.iter().zip(exp.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>8.2}x | {:>6}",
            "6. f64 (Sensores K=64 +/-)",
            lat_ms,
            std_ms,
            std_ms / lat_ms,
            eq
        );
        assert!(eq);
    }

    println!("\n=== ETAPA 4: EXECUCAO PARALELA MULTI-CORE (par_lattice_sort) ===");
    println!(
        "{:<34} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Cenario Uniform 32-bit", "Par (ms)", "1-Thr (ms)", "Std (ms)", "Par vs Std"
    );
    println!("{}", "-".repeat(90));

    for &(label, size) in &[
        ("Uniform 32-bit (N = 1,000,000)", 1_000_000_usize),
        ("Uniform 32-bit (N = 10,000,000)", 10_000_000_usize),
    ] {
        let mut seed = 314_159_265_u32;
        let data: Vec<i32> = (0..size)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed as i32
            })
            .collect();
        let (std_ms, exp) = median_ms(&data, <[i32]>::sort_unstable, 3);
        let (seq_ms, seq_act) = median_ms(&data, candidate::lattice_sort_i32, 3);
        let (par_ms, par_act) = median_ms(&data, par_lattice_sort, 3);
        assert_eq!(seq_act, exp);
        assert_eq!(par_act, exp);
        println!(
            "{:<34} | {:>12.2} | {:>12.2} | {:>12.2} | {:>9.2}x",
            label,
            par_ms,
            seq_ms,
            std_ms,
            std_ms / par_ms
        );
    }
}
