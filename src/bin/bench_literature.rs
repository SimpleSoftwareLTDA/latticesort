//! Empirical benchmark comparing `latticesort` against well-known sorting algorithms from
//! Computer Science literature at $N = 1{,}000{,}000$:
//! 1. `ipnsort` (`std::slice::sort_unstable`, Bergdoll & Peters 2024)
//! 2. `driftsort` (`std::slice::sort`, Bergdoll & Peters 2024)
//! 3. Classic 4-Pass LSD Radix Sort (Seward 1954 / Knuth 1973, separate histograms, no skip)
//! 4. American Flag Sort / In-Place MSB Radix Sort (McIlroy, Bostic & McIlroy 1993)
//! 5. Bentley-McIlroy 3-Way Quicksort (Bentley & McIlroy 1993)
//! 6. Floyd Bottom-Up HeapSort (Floyd 1964)

use latticesort::{candidate, harness};
use std::time::Instant;

struct LiteratureAlgorithm {
    name: &'static str,
    citation: &'static str,
    sort_fn: fn(&mut [i32]),
}

struct LiteratureDataset {
    label: &'static str,
    values: Vec<i32>,
}

/// Classic 4-Pass LSD Radix Sort (Seward 1954 / Knuth Vol. 3).
///
/// Computes each 256-entry histogram immediately before its scatter pass (4 histogram passes +
/// 4 scatter passes = 8 memory passes total), without degenerate-byte skipping or L1 fast paths.
fn classic_lsd_radix_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    let mut scratch = vec![0_i32; len];
    for pass in 0..4 {
        let shift = pass * 8;
        let flip = pass == 3;
        let src: &[i32] = if pass % 2 == 0 { slice } else { &scratch };
        let mut hist = [0_usize; 256];
        for &x in src {
            let raw = (x as u32) >> shift;
            let bucket = if flip {
                ((raw ^ 0x80) & 0xff) as usize
            } else {
                (raw & 0xff) as usize
            };
            hist[bucket] += 1;
        }
        let mut offsets = [0_usize; 256];
        let mut sum = 0_usize;
        let indexed_hist = hist.iter().enumerate();
        for (i, &count) in indexed_hist {
            offsets[i] = sum;
            sum += count;
        }
        if pass % 2 == 0 {
            for &x in slice.iter() {
                let raw = (x as u32) >> shift;
                let bucket = if flip {
                    ((raw ^ 0x80) & 0xff) as usize
                } else {
                    (raw & 0xff) as usize
                };
                scratch[offsets[bucket]] = x;
                offsets[bucket] += 1;
            }
        } else {
            for &x in scratch.iter() {
                let raw = (x as u32) >> shift;
                let bucket = if flip {
                    ((raw ^ 0x80) & 0xff) as usize
                } else {
                    (raw & 0xff) as usize
                };
                slice[offsets[bucket]] = x;
                offsets[bucket] += 1;
            }
        }
    }
}

/// American Flag Sort (McIlroy, Bostic & McIlroy 1993).
///
/// In-place top-down 256-ary MSB radix sort by cyclic permutation of buckets.
fn american_flag_sort(slice: &mut [i32]) {
    fn insertion_fallback(arr: &mut [i32]) {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
    }

    fn flag_recurse(arr: &mut [i32], shift: usize, flip_sign: bool) {
        if arr.len() <= 32 {
            insertion_fallback(arr);
            return;
        }
        let mut counts = [0_usize; 256];
        for &x in arr.iter() {
            let raw = (x as u32) >> shift;
            let b = if flip_sign {
                ((raw ^ 0x80) & 0xff) as usize
            } else {
                (raw & 0xff) as usize
            };
            counts[b] += 1;
        }
        let mut starts = [0_usize; 256];
        let mut ends = [0_usize; 256];
        let mut sum = 0_usize;
        for b in 0..256 {
            starts[b] = sum;
            sum += counts[b];
            ends[b] = sum;
        }
        let mut cursors = starts;
        for b in 0..256 {
            while cursors[b] < ends[b] {
                let val = arr[cursors[b]];
                let raw = (val as u32) >> shift;
                let target_b = if flip_sign {
                    ((raw ^ 0x80) & 0xff) as usize
                } else {
                    (raw & 0xff) as usize
                };
                if target_b == b {
                    cursors[b] += 1;
                } else {
                    let dst_idx = cursors[target_b];
                    arr.swap(cursors[b], dst_idx);
                    cursors[target_b] += 1;
                }
            }
        }
        if shift > 0 {
            for b in 0..256 {
                if counts[b] > 1 {
                    flag_recurse(&mut arr[starts[b]..ends[b]], shift - 8, false);
                }
            }
        }
    }

    if slice.len() > 1 {
        flag_recurse(slice, 24, true);
    }
}

/// Bentley-McIlroy 3-Way Partitioning Quicksort (Bentley & McIlroy 1993).
///
/// Median-of-three pivot + Dutch National Flag 3-way partition (`< pivot`, `== pivot`, `> pivot`).
fn bentley_mcilroy_quicksort(slice: &mut [i32]) {
    if slice.len() <= 24 {
        for i in 1..slice.len() {
            let key = slice[i];
            let mut j = i;
            while j > 0 && slice[j - 1] > key {
                slice[j] = slice[j - 1];
                j -= 1;
            }
            slice[j] = key;
        }
        return;
    }
    let len = slice.len();
    let mid = len / 2;
    if slice[mid] < slice[0] {
        slice.swap(0, mid);
    }
    if slice[len - 1] < slice[0] {
        slice.swap(0, len - 1);
    }
    if slice[len - 1] < slice[mid] {
        slice.swap(mid, len - 1);
    }
    let pivot = slice[mid];

    let mut lt = 0_usize;
    let mut i = 0_usize;
    let mut gt = len;
    while i < gt {
        if slice[i] < pivot {
            slice.swap(lt, i);
            lt += 1;
            i += 1;
        } else if slice[i] > pivot {
            gt -= 1;
            slice.swap(i, gt);
        } else {
            i += 1;
        }
    }
    bentley_mcilroy_quicksort(&mut slice[..lt]);
    bentley_mcilroy_quicksort(&mut slice[gt..]);
}

/// Floyd HeapSort (Williams & Floyd 1964).
///
/// Classic $O(n \log n)$ in-place binary heap sort.
fn floyd_heapsort(slice: &mut [i32]) {
    fn sift_down(arr: &mut [i32], mut root: usize, end: usize) {
        loop {
            let mut child = 2 * root + 1;
            if child >= end {
                break;
            }
            if child + 1 < end && arr[child] < arr[child + 1] {
                child += 1;
            }
            if arr[root] < arr[child] {
                arr.swap(root, child);
                root = child;
            } else {
                break;
            }
        }
    }

    let len = slice.len();
    if len <= 1 {
        return;
    }
    for start in (0..len / 2).rev() {
        sift_down(slice, start, len);
    }
    for end in (1..len).rev() {
        slice.swap(0, end);
        sift_down(slice, 0, end);
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

fn main() {
    let n = 1_000_000;
    let datasets: Vec<LiteratureDataset> = vec![
        LiteratureDataset {
            label: "1. Cyclic K=12",
            values: harness::make_mod_k(n, 12),
        },
        LiteratureDataset {
            label: "2. Cyclic K=13",
            values: harness::make_mod_k(n, 13),
        },
        LiteratureDataset {
            label: "3. Cyclic K=17 (prime)",
            values: harness::make_mod_k(n, 17),
        },
        LiteratureDataset {
            label: "4. Wide K=13 (span 10^9)",
            values: harness::make_wide_mod_k(n, 13),
        },
        LiteratureDataset {
            label: "5. Random K=100",
            values: harness::make_random_mod_k(n, 100),
        },
        LiteratureDataset {
            label: "6. Random K=1000",
            values: harness::make_random_mod_k(n, 1000),
        },
        LiteratureDataset {
            label: "7. Bitonic Soliton Wave",
            values: harness::make_soliton_wave(n),
        },
        LiteratureDataset {
            label: "8. Pareto Power-Law",
            values: harness::make_pareto_power_law(n),
        },
        LiteratureDataset {
            label: "9. Uniform 32-bit Random",
            values: harness::make_uniform_random(n),
        },
    ];

    let algorithms = [
        LiteratureAlgorithm {
            name: "ipnsort (std::slice::sort_unstable)",
            citation: "Bergdoll & Peters, 2024 (Rust std)",
            sort_fn: <[i32]>::sort_unstable,
        },
        LiteratureAlgorithm {
            name: "driftsort (std::slice::sort)",
            citation: "Bergdoll & Peters, 2024 (Rust std stable)",
            sort_fn: <[i32]>::sort,
        },
        LiteratureAlgorithm {
            name: "Classic 4-Pass LSD Radix Sort",
            citation: "Seward 1954 / Knuth TAOCP Vol. 3",
            sort_fn: classic_lsd_radix_sort,
        },
        LiteratureAlgorithm {
            name: "American Flag Sort (In-Place MSB Radix)",
            citation: "McIlroy, Bostic & McIlroy, 1993",
            sort_fn: american_flag_sort,
        },
        LiteratureAlgorithm {
            name: "Bentley-McIlroy 3-Way Quicksort",
            citation: "Bentley & McIlroy, 1993 (C qsort)",
            sort_fn: bentley_mcilroy_quicksort,
        },
        LiteratureAlgorithm {
            name: "Floyd HeapSort",
            citation: "Williams & Floyd, 1964",
            sort_fn: floyd_heapsort,
        },
    ];

    let mut work_lat = vec![0_i32; n];
    let mut work_other = vec![0_i32; n];

    let mut lat_log_sum = 0.0_f64;
    for dataset in &datasets {
        let ms = measure_median_ms(
            &dataset.values,
            &mut work_lat,
            candidate::lattice_sort_i32,
            3,
        );
        lat_log_sum += ms.ln();
    }
    let lat_geomean = (lat_log_sum / datasets.len() as f64).exp();

    println!("=== LATTICESORT vs. ALGORITMOS DA LITERATURA (N = {n}) ===");
    println!("LatticeSort Geometric Mean (9 datasets): {lat_geomean:.3} ms\n");
    println!(
        "{:<40} | {:<36} | {:>10} | {:>10} | {:>10}",
        "Algoritmo da Literatura", "Referencia", "Geomean", "Uniform32", "Speedup"
    );
    println!("{}", "-".repeat(118));

    for algo in &algorithms {
        let mut log_sum = 0.0_f64;
        let mut uniform_ms = 0.0_f64;
        let indexed_datasets = datasets.iter().enumerate();
        for (idx, dataset) in indexed_datasets {
            let ms = measure_median_ms(&dataset.values, &mut work_other, algo.sort_fn, 3);
            log_sum += ms.ln();
            if idx == 8 {
                uniform_ms = ms;
            }
            if idx == 0 {
                work_lat.copy_from_slice(&dataset.values);
                candidate::lattice_sort_i32(&mut work_lat);
                assert_eq!(
                    work_other, work_lat,
                    "{} failed correctness check on {}",
                    algo.name, dataset.label
                );
            }
        }
        let geomean = (log_sum / datasets.len() as f64).exp();
        let speedup_geomean = geomean / lat_geomean;
        println!(
            "{:<40} | {:<36} | {:>7.3} ms | {:>7.3} ms | {:>9.2}x",
            algo.name, algo.citation, geomean, uniform_ms, speedup_geomean
        );
    }
}
