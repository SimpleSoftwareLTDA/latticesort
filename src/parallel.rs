//! Multi-core parallel sorting scheduler (`parallel_lattice_sort`).
//!
//! Uses scoped threads (`std::thread::scope`) and disjoint mutable slice splitting
//! (`split_at_mut`) to execute both the top-level MSB scatter and the per-bucket radix passes
//! across all CPU cores without `unsafe` blocks.

use crate::candidate::{self, LatticeKey32};
use crate::kernel_i64::{self, LatticeKey64};
use std::num::NonZeroUsize;
use std::thread;

struct DisjointBucketTask<'slice, T> {
    bucket_index: usize,
    target_bucket: &'slice mut [T],
}

/// Sorts a mutable slice of any 32-bit `T: LatticeKey32` (`i32`, `u32`, `f32`) in parallel
/// across available CPU cores.
pub fn parallel_lattice_sort_32<T: LatticeKey32>(slice: &mut [T]) {
    let len = slice.len();
    if len < 200_000 {
        candidate::lattice_sort_generic_32(slice);
        return;
    }

    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(64);
    if threads <= 1 {
        candidate::lattice_sort_generic_32(slice);
        return;
    }

    if candidate::try_bitonic_run_merge_generic(slice) {
        return;
    }
    let outcome = candidate::try_sentinel_hash_aggregation_generic(slice);
    if outcome.sorted {
        return;
    }
    if !outcome.span_exceeds_horizon && candidate::try_dense_range_count_generic(slice) {
        return;
    }

    let chunk_size = len.div_ceil(threads);
    let num_chunks = slice.chunks(chunk_size).len();
    let mut local_counts = vec![[0_usize; 256]; num_chunks];
    let mut local_offsets = vec![[0_usize; 256]; num_chunks];
    let mut scratch = vec![T::ZERO; len];

    // Phase 1: Parallel MSB histogram + parallel chunk-local MSB scatter into `scratch`.
    thread::scope(|scope| {
        let slice_chunks = slice.chunks(chunk_size);
        let scratch_chunks = scratch.chunks_mut(chunk_size);
        let chunk_iter = slice_chunks
            .zip(scratch_chunks)
            .zip(local_counts.iter_mut())
            .zip(local_offsets.iter_mut());

        for (((src_chunk, dst_chunk), counts_out), offsets_out) in chunk_iter {
            scope.spawn(move || {
                let (chunks4, remainder) = src_chunk.as_chunks::<4>();
                for block in chunks4 {
                    counts_out[(((block[0].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
                    counts_out[(((block[1].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
                    counts_out[(((block[2].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
                    counts_out[(((block[3].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
                }
                for &item in remainder {
                    counts_out[(((item.to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
                }

                let mut running_offset = 0_usize;
                for b in 0..256 {
                    offsets_out[b] = running_offset;
                    running_offset += counts_out[b];
                }

                let mut write_cursors = *offsets_out;
                for &item in src_chunk {
                    let b = (((item.to_key() as u32) >> 24) ^ 0x80) as usize;
                    dst_chunk[write_cursors[b]] = item;
                    write_cursors[b] += 1;
                }
            });
        }
    });

    let mut msb_histogram = [0_usize; 256];
    for chunk_counts in &local_counts {
        for b in 0..256 {
            msb_histogram[b] += chunk_counts[b];
        }
    }

    if msb_histogram.contains(&len) {
        candidate::l2_local_3pass_radix_sort(slice, &mut scratch);
        return;
    }

    let mut tasks: Vec<DisjointBucketTask<'_, T>> = Vec::with_capacity(256);
    let mut remaining_slice = &mut slice[..];
    let indexed_msb = msb_histogram.iter().enumerate();
    for (bucket_index, &count) in indexed_msb {
        if count == 0 {
            continue;
        }
        let (slice_head, slice_tail) = remaining_slice.split_at_mut(count);
        remaining_slice = slice_tail;
        tasks.push(DisjointBucketTask {
            bucket_index,
            target_bucket: slice_head,
        });
    }

    // Phase 2: Parallel gathering of each MSB bucket from `scratch` + local lower-24-bit sort.
    let target_per_thread = len.div_ceil(threads);
    let scratch_ref = &scratch[..];
    let counts_ref = &local_counts[..];
    let offsets_ref = &local_offsets[..];

    thread::scope(|scope| {
        let mut remaining_tasks = &mut tasks[..];
        while !remaining_tasks.is_empty() {
            let mut accumulated_elements = 0_usize;
            let mut split_idx = 0_usize;
            let indexed_tasks = remaining_tasks.iter().enumerate();
            for (i, task) in indexed_tasks {
                accumulated_elements += task.target_bucket.len();
                split_idx = i + 1;
                if accumulated_elements >= target_per_thread {
                    break;
                }
            }
            let (batch, rest) = remaining_tasks.split_at_mut(split_idx);
            remaining_tasks = rest;
            scope.spawn(move || {
                let max_bucket_len = batch
                    .iter()
                    .map(|task| task.target_bucket.len())
                    .max()
                    .unwrap_or(0);
                let mut local_scratch = vec![T::ZERO; max_bucket_len];

                for task in batch {
                    let b = task.bucket_index;
                    let total_count = task.target_bucket.len();
                    let bucket_scratch = &mut local_scratch[..total_count];
                    let mut dst_pos = 0_usize;
                    for chunk_idx in 0..num_chunks {
                        let part_len = counts_ref[chunk_idx][b];
                        if part_len > 0 {
                            let src_start = chunk_idx * chunk_size + offsets_ref[chunk_idx][b];
                            bucket_scratch[dst_pos..dst_pos + part_len]
                                .copy_from_slice(&scratch_ref[src_start..src_start + part_len]);
                            dst_pos += part_len;
                        }
                    }

                    if total_count == 1 {
                        task.target_bucket[0] = bucket_scratch[0];
                    } else {
                        candidate::l2_local_3pass_radix_sort(task.target_bucket, bucket_scratch);
                    }
                }
            });
        }
    });
}

/// Sorts a mutable slice of any 64-bit `T: LatticeKey64` (`i64`, `u64`, `f64`) in parallel
/// across available CPU cores.
pub fn parallel_lattice_sort_64<T: LatticeKey64>(slice: &mut [T]) {
    let len = slice.len();
    if len < 200_000 {
        kernel_i64::lattice_sort_generic_64(slice);
        return;
    }

    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(64);
    if threads <= 1 {
        kernel_i64::lattice_sort_generic_64(slice);
        return;
    }

    if kernel_i64::try_bitonic_run_merge_64(slice) {
        return;
    }
    let outcome = kernel_i64::try_sentinel_hash_aggregation_64(slice);
    if outcome.sorted {
        return;
    }
    if !outcome.span_exceeds_horizon && kernel_i64::try_dense_range_count_64(slice) {
        return;
    }

    let chunk_size = len.div_ceil(threads);
    let num_chunks = slice.chunks(chunk_size).len();
    let mut local_counts = vec![[0_usize; 256]; num_chunks];
    let mut local_offsets = vec![[0_usize; 256]; num_chunks];
    let mut scratch = vec![T::ZERO; len];

    thread::scope(|scope| {
        let slice_chunks = slice.chunks(chunk_size);
        let scratch_chunks = scratch.chunks_mut(chunk_size);
        let chunk_iter = slice_chunks
            .zip(scratch_chunks)
            .zip(local_counts.iter_mut())
            .zip(local_offsets.iter_mut());

        for (((src_chunk, dst_chunk), counts_out), offsets_out) in chunk_iter {
            scope.spawn(move || {
                for &item in src_chunk {
                    counts_out[(((item.to_key() as u64) >> 56) ^ 0x80) as usize] += 1;
                }

                let mut running_offset = 0_usize;
                for b in 0..256 {
                    offsets_out[b] = running_offset;
                    running_offset += counts_out[b];
                }

                let mut write_cursors = *offsets_out;
                for &item in src_chunk {
                    let b = (((item.to_key() as u64) >> 56) ^ 0x80) as usize;
                    dst_chunk[write_cursors[b]] = item;
                    write_cursors[b] += 1;
                }
            });
        }
    });

    let mut msb_histogram = [0_usize; 256];
    for chunk_counts in &local_counts {
        for b in 0..256 {
            msb_histogram[b] += chunk_counts[b];
        }
    }

    if msb_histogram.contains(&len) {
        kernel_i64::radix_lsd_fused_sort_64(slice);
        return;
    }

    let mut tasks: Vec<DisjointBucketTask<'_, T>> = Vec::with_capacity(256);
    let mut remaining_slice = &mut slice[..];
    let indexed_msb = msb_histogram.iter().enumerate();
    for (bucket_index, &count) in indexed_msb {
        if count == 0 {
            continue;
        }
        let (slice_head, slice_tail) = remaining_slice.split_at_mut(count);
        remaining_slice = slice_tail;
        tasks.push(DisjointBucketTask {
            bucket_index,
            target_bucket: slice_head,
        });
    }

    let target_per_thread = len.div_ceil(threads);
    let scratch_ref = &scratch[..];
    let counts_ref = &local_counts[..];
    let offsets_ref = &local_offsets[..];

    thread::scope(|scope| {
        let mut remaining_tasks = &mut tasks[..];
        while !remaining_tasks.is_empty() {
            let mut accumulated_elements = 0_usize;
            let mut split_idx = 0_usize;
            let indexed_tasks = remaining_tasks.iter().enumerate();
            for (i, task) in indexed_tasks {
                accumulated_elements += task.target_bucket.len();
                split_idx = i + 1;
                if accumulated_elements >= target_per_thread {
                    break;
                }
            }
            let (batch, rest) = remaining_tasks.split_at_mut(split_idx);
            remaining_tasks = rest;
            scope.spawn(move || {
                for task in batch {
                    let b = task.bucket_index;
                    let mut dst_pos = 0_usize;
                    for chunk_idx in 0..num_chunks {
                        let part_len = counts_ref[chunk_idx][b];
                        if part_len > 0 {
                            let src_start = chunk_idx * chunk_size + offsets_ref[chunk_idx][b];
                            task.target_bucket[dst_pos..dst_pos + part_len]
                                .copy_from_slice(&scratch_ref[src_start..src_start + part_len]);
                            dst_pos += part_len;
                        }
                    }

                    if task.target_bucket.len() > 1 {
                        kernel_i64::radix_lsd_fused_sort_64(task.target_bucket);
                    }
                }
            });
        }
    });
}

/// Sorts a mutable slice of `i32` in parallel across available CPU cores.
#[inline]
pub fn parallel_lattice_sort(slice: &mut [i32]) {
    parallel_lattice_sort_32(slice);
}
