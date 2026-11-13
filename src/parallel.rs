//! Multi-core parallel sorting scheduler (`parallel_lattice_sort`).
//!
//! Uses scoped threads (`std::thread::scope`) and disjoint mutable slice splitting
//! (`split_at_mut`) to execute MSB bucket sorting across CPU cores without `unsafe` blocks.

use crate::candidate;
use std::num::NonZeroUsize;
use std::thread;

struct BucketWorkPair<'slice> {
    target_bucket: &'slice mut [i32],
    scratch_bucket: &'slice mut [i32],
}

/// Sorts a mutable slice of `i32` in parallel across available CPU cores.
///
/// Falls back to sequential `candidate::lattice_sort_i32` for slices smaller than `200_000`
/// elements or when a fast-path stage (`try_bitonic_run_merge`, `try_sentinel_hash_aggregation`,
/// or `try_dense_range_count`) resolves the slice in a single pass.
pub fn parallel_lattice_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len < 200_000 {
        candidate::lattice_sort_i32(slice);
        return;
    }

    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(16);
    if threads <= 1 {
        candidate::lattice_sort_i32(slice);
        return;
    }

    if candidate::try_bitonic_run_merge(slice) {
        return;
    }
    let outcome = candidate::try_sentinel_hash_aggregation(slice);
    if outcome.sorted {
        return;
    }
    if !outcome.span_exceeds_horizon && candidate::try_dense_range_count(slice) {
        return;
    }

    let chunk_size = len.div_ceil(threads);
    let mut local_histograms = vec![[0_u32; 256]; threads];
    thread::scope(|scope| {
        let chunk_pairs = slice.chunks(chunk_size).zip(local_histograms.iter_mut());
        for (chunk, histogram_out) in chunk_pairs {
            scope.spawn(move || {
                let (chunks4, remainder) = chunk.as_chunks::<4>();
                for block in chunks4 {
                    histogram_out[(((block[0] as u32) >> 24) ^ 0x80) as usize] += 1;
                    histogram_out[(((block[1] as u32) >> 24) ^ 0x80) as usize] += 1;
                    histogram_out[(((block[2] as u32) >> 24) ^ 0x80) as usize] += 1;
                    histogram_out[(((block[3] as u32) >> 24) ^ 0x80) as usize] += 1;
                }
                for &x in remainder {
                    histogram_out[(((x as u32) >> 24) ^ 0x80) as usize] += 1;
                }
            });
        }
    });

    let mut msb_histogram = [0_u32; 256];
    for local_histogram in &local_histograms {
        for bucket in 0..256 {
            msb_histogram[bucket] += local_histogram[bucket];
        }
    }

    let mut scratch = vec![0_i32; len];
    let n32 = len as u32;
    if msb_histogram.contains(&n32) {
        scratch.copy_from_slice(slice);
        candidate::l2_local_3pass_radix_sort(slice, &mut scratch);
        return;
    }

    candidate::scatter_pass::<24, true>(slice, &mut scratch, &msb_histogram);

    let mut buckets: Vec<BucketWorkPair<'_>> = Vec::with_capacity(256);
    let mut remaining_slice = &mut slice[..];
    let mut remaining_scratch = &mut scratch[..];
    for &count_u32 in &msb_histogram {
        let count = count_u32 as usize;
        if count == 0 {
            continue;
        }
        let (slice_head, slice_tail) = remaining_slice.split_at_mut(count);
        let (scratch_head, scratch_tail) = remaining_scratch.split_at_mut(count);
        remaining_slice = slice_tail;
        remaining_scratch = scratch_tail;
        buckets.push(BucketWorkPair {
            target_bucket: slice_head,
            scratch_bucket: scratch_head,
        });
    }

    let target_per_thread = len.div_ceil(threads);
    thread::scope(|scope| {
        let mut remaining_buckets = &mut buckets[..];
        while !remaining_buckets.is_empty() {
            let mut accumulated_elements = 0_usize;
            let mut split_idx = 0_usize;
            let indexed_buckets = remaining_buckets.iter().enumerate();
            for (i, pair) in indexed_buckets {
                accumulated_elements += pair.target_bucket.len();
                split_idx = i + 1;
                if accumulated_elements >= target_per_thread {
                    break;
                }
            }
            let (batch, rest) = remaining_buckets.split_at_mut(split_idx);
            remaining_buckets = rest;
            scope.spawn(move || {
                for pair in batch {
                    let count = pair.target_bucket.len();
                    if count == 1 {
                        pair.target_bucket[0] = pair.scratch_bucket[0];
                    } else {
                        candidate::l2_local_3pass_radix_sort(
                            pair.target_bucket,
                            pair.scratch_bucket,
                        );
                    }
                }
            });
        }
    });
}
