//! 32-bit adaptive sorting kernel for `latticesort`.
//!
//! Combines four specialized stages generic over any 32-bit bijective key type (`i32`, `u32`, `f32`):
//! 1. `try_bitonic_run_merge`: linear-time $O(n)$ run detection and merge for monotonic,
//!    bitonic, and mostly-sorted inputs (stack-buffered for $N \le 2048$).
//! 2. `try_sentinel_hash_aggregation`: L1-resident 1024-slot dual-sentinel hash table for
//!    low-cardinality inputs (up to 384 distinct keys).
//! 3. `try_dense_range_count`: offset counting sort when the key span fits within cache.
//! 4. `radix_lsd_fused_sort` and `radix_msb_l2_hybrid_sort`: 4-pass 8-bit radix sort with
//!    single-pass fused histograms and degenerate-byte skipping.

const DENSE_RANGE_MAX_SPAN: u64 = 1_048_576;
pub(crate) const SMALL_SORT_CUTOFF: usize = 384;
pub(crate) const STACK_SCRATCH_MAX: usize = 2048;
const MSB_L2_PARTITION_THRESHOLD: usize = 1_500_000;
const MSB_ENTROPY_GATE_MIN_LEN: usize = 200_000;

const SENTINEL_HASH_MULTIPLIERS_32: [u32; 4] = [0x9e37_79b9, 0x85eb_ca6b, 0xc2b2_ae35, 0x27d4_eb2f];

/// Bijective 32-bit key mapping onto the signed `i32` total-order lattice.
pub trait LatticeKey32: Copy + Send + Sync {
    /// Zero-initialized default value used for scratch buffers.
    const ZERO: Self;

    /// Projects `self` onto `i32` preserving total order ($a \le b \iff \text{to\_key}(a) \le \text{to\_key}(b)$).
    fn to_key(self) -> i32;

    /// Inverse bijection reconstructing `Self` from its `i32` lattice key.
    fn from_key(key: i32) -> Self;

    /// Sorts a small slice ($\le 384$ elements) in-place without heap allocation.
    fn sort_small_in_place(slice: &mut [Self]);
}

impl LatticeKey32 for i32 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn to_key(self) -> i32 {
        self
    }

    #[inline(always)]
    fn from_key(key: i32) -> Self {
        key
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable();
    }
}

impl LatticeKey32 for u32 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn to_key(self) -> i32 {
        (self ^ 0x8000_0000) as i32
    }

    #[inline(always)]
    fn from_key(key: i32) -> Self {
        (key as u32) ^ 0x8000_0000
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable();
    }
}

impl LatticeKey32 for f32 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn to_key(self) -> i32 {
        let bits = self.to_bits();
        let mask = (((bits as i32) >> 31) as u32) & 0x7fff_ffff;
        (bits ^ mask) as i32
    }

    #[inline(always)]
    fn from_key(key: i32) -> Self {
        let bits = key as u32;
        let mask = ((key >> 31) as u32) & 0x7fff_ffff;
        Self::from_bits(bits ^ mask)
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable_by(Self::total_cmp);
    }
}

/// Outcome of the dual-sentinel L1 hash aggregation probe and pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub struct HashAggregationOutcome {
    /// Whether the slice was completely sorted by the hash aggregation pass.
    pub sorted: bool,
    /// Whether the sampled value span exceeds the dense range counting horizon.
    pub span_exceeds_horizon: bool,
}

#[derive(Clone, Copy)]
struct KeyCount32 {
    key: i32,
    count: usize,
}

fn insertion_sort_i32(slice: &mut [i32]) {
    for i in 1..slice.len() {
        let key = slice[i];
        let mut j = i;
        while j > 0 && slice[j - 1] > key {
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = key;
    }
}

fn merge_sorted_halves_generic<T: LatticeKey32>(slice: &mut [T], split: usize) {
    if slice[split - 1].to_key() <= slice[split].to_key() {
        return;
    }
    let first_val = slice[0];
    let first_key = first_val.to_key();
    if first_key == slice[split - 1].to_key() {
        let insert_pos = slice[split..].partition_point(|x| x.to_key() < first_key);
        slice.copy_within(split..split + insert_pos, 0);
        slice[insert_pos..split + insert_pos].fill(first_val);
        return;
    }

    let right_first_key = slice[split].to_key();
    let trim_left = slice[..split].partition_point(|x| x.to_key() <= right_first_key);
    let active = &mut slice[trim_left..];
    let adjusted_split = split - trim_left;

    let len = active.len();
    let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX];
    let mut heap_buf: Vec<T>;
    let merged: &mut [T] = if len <= STACK_SCRATCH_MAX {
        &mut stack_buf[..len]
    } else {
        heap_buf = vec![T::ZERO; len];
        &mut heap_buf
    };

    let (left, right) = active.split_at(adjusted_split);
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut k = 0_usize;
    while i < left.len() && j < right.len() {
        let a = left[i];
        let b = right[j];
        if a.to_key() <= b.to_key() {
            merged[k] = a;
            i += 1;
        } else {
            merged[k] = b;
            j += 1;
        }
        k += 1;
    }
    if i < left.len() {
        let remaining = left.len() - i;
        merged[k..k + remaining].copy_from_slice(&left[i..]);
    } else if j < right.len() {
        let remaining = right.len() - j;
        merged[k..k + remaining].copy_from_slice(&right[j..]);
    }
    active.copy_from_slice(merged);
}

/// Generic linear-time bitonic and structural run merger for any `T: LatticeKey32`.
#[must_use]
pub(crate) fn try_bitonic_run_merge_generic<T: LatticeKey32>(slice: &mut [T]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
        return false;
    }

    let step = len / 8;
    let mut dir_changes = 0_usize;
    let mut prev_dir = 0_i8;
    for i in 1..8 {
        let a = slice[(i - 1) * step].to_key();
        let b = slice[i * step].to_key();
        let dir = if b > a {
            1_i8
        } else if b < a {
            -1_i8
        } else {
            prev_dir
        };
        if prev_dir != 0 && dir != 0 && dir != prev_dir {
            dir_changes += 1;
        }
        if dir != 0 {
            prev_dir = dir;
        }
    }
    if dir_changes > 1 {
        return false;
    }

    let mut peak = 0_usize;
    while peak + 1 < len && slice[peak].to_key() <= slice[peak + 1].to_key() {
        peak += 1;
    }
    if peak == len - 1 {
        return true;
    }

    if peak == 0 {
        let mut valley = 0_usize;
        while valley + 1 < len && slice[valley].to_key() >= slice[valley + 1].to_key() {
            valley += 1;
        }
        if valley == len - 1 {
            slice.reverse();
            return true;
        }
        let mut tail = valley + 1;
        while tail + 1 < len && slice[tail].to_key() <= slice[tail + 1].to_key() {
            tail += 1;
        }
        if tail == len - 1 {
            slice[..=valley].reverse();
            merge_sorted_halves_generic(slice, valley + 1);
            return true;
        }
        return false;
    }

    let mut tail = peak + 1;
    while tail + 1 < len && slice[tail].to_key() >= slice[tail + 1].to_key() {
        tail += 1;
    }
    if tail == len - 1 {
        let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX];
        let mut heap_buf: Vec<T>;
        let merged: &mut [T] = if len <= STACK_SCRATCH_MAX {
            &mut stack_buf[..len]
        } else {
            heap_buf = vec![T::ZERO; len];
            &mut heap_buf
        };

        let (left, right) = slice.split_at(peak + 1);
        let mut i = 0_usize;
        let mut j = right.len();
        let mut k = 0_usize;
        while i < left.len() && j > 0 {
            let a = left[i];
            let b = right[j - 1];
            if a.to_key() <= b.to_key() {
                merged[k] = a;
                i += 1;
            } else {
                merged[k] = b;
                j -= 1;
            }
            k += 1;
        }
        if i < left.len() {
            let remaining = left.len() - i;
            merged[k..k + remaining].copy_from_slice(&left[i..]);
        } else {
            while j > 0 {
                merged[k] = right[j - 1];
                j -= 1;
                k += 1;
            }
        }
        slice.copy_from_slice(merged);
        return true;
    }

    if peak >= (len * 3) / 4 {
        let first_key = slice[0].to_key();
        if slice[peak / 2].to_key() == first_key {
            while peak > 0 && slice[peak].to_key() != first_key {
                peak -= 1;
            }
        }
        lattice_sort_generic_32(&mut slice[peak + 1..]);
        merge_sorted_halves_generic(slice, peak + 1);
        return true;
    }

    false
}

/// Detects and merges monotonic, bitonic, or mostly-sorted `i32` runs in linear time.
#[must_use]
pub fn try_bitonic_run_merge(slice: &mut [i32]) -> bool {
    try_bitonic_run_merge_generic(slice)
}

#[inline]
fn sentinel_hash_1024(val: i32, multiplier: u32) -> usize {
    let u = val as u32;
    (((u ^ (u >> 15)).wrapping_mul(multiplier)) >> 22) as usize
}

/// Generic 1024-slot dual-sentinel L1 hash aggregation for any `T: LatticeKey32`.
///
/// Incorporates two memory-layout optimizations inspired by Cloudflare's 1.1.1.1 cache design:
/// 1. **Bitpacked Occupancy (`[u64; 16]` = 128 B instead of `[bool; 1024]` = 1024 B)**:
///    Shrinks the occupancy table by 8x (to 2 cache lines) and enumerates active slots in
///    $O(K)$ hardware `tzcnt` (`trailing_zeros`) steps instead of scanning all 1024 slots.
/// 2. **Wire-Packed 64-Bit `(key, count)` Entries (`u64` instead of 16-byte padded struct)**:
///    Packs the order-preserving unsigned key in the upper 32 bits and the count in the lower
///    32 bits, eliminating 4 bytes of struct alignment padding per entry (50% smaller footprint).
pub(crate) fn try_sentinel_hash_aggregation_generic<T: LatticeKey32>(
    slice: &mut [T],
) -> HashAggregationOutcome {
    let len = slice.len();
    if len <= STACK_SCRATCH_MAX {
        return HashAggregationOutcome {
            sorted: false,
            span_exceeds_horizon: true,
        };
    }

    let mut probe = [0_i32; 128];
    let anchors = [len / 10, (len * 3) / 10, len / 2, (len * 7) / 10, len - 16];
    let indexed_anchors = anchors.iter().enumerate();
    for (window_idx, &base) in indexed_anchors {
        let dst_start = window_idx * 16;
        for k in 0..16 {
            probe[dst_start + k] = slice[base + k].to_key();
        }
    }

    let stride = ((len as u64 * 2_654_435_761) >> 32).max(1) as usize | 1;
    let mut idx = len / 7;
    for slot in &mut probe[80..] {
        *slot = slice[idx % len].to_key();
        idx = idx.wrapping_add(stride);
    }
    insertion_sort_i32(&mut probe);

    let probe_span = (i64::from(probe[127]) - i64::from(probe[0])) as u64;
    let horizon = (len as u64).min(DENSE_RANGE_MAX_SPAN);
    let span_exceeds_horizon = probe_span > horizon;

    let mut unique_probe = [0_i32; 128];
    unique_probe[0] = probe[0];
    let mut unique_probe_len = 1_usize;
    for i in 1..128 {
        if probe[i] != probe[i - 1] {
            if unique_probe_len >= 80 {
                return HashAggregationOutcome {
                    sorted: false,
                    span_exceeds_horizon,
                };
            }
            unique_probe[unique_probe_len] = probe[i];
            unique_probe_len += 1;
        }
    }

    let mut chosen_multiplier = SENTINEL_HASH_MULTIPLIERS_32[0];
    for &multiplier in &SENTINEL_HASH_MULTIPLIERS_32 {
        let mut occupied_bits = [0_u64; 16];
        let mut collision_free = true;
        for &key in &unique_probe[..unique_probe_len] {
            let slot = sentinel_hash_1024(key, multiplier);
            let word_idx = slot >> 6;
            let mask = 1_u64 << (slot & 63);
            if (occupied_bits[word_idx] & mask) != 0 {
                collision_free = false;
                break;
            }
            occupied_bits[word_idx] |= mask;
        }
        if collision_free {
            chosen_multiplier = multiplier;
            break;
        }
    }

    let sentinel_zero = 0_i32;
    let home_slot_zero = sentinel_hash_1024(sentinel_zero, chosen_multiplier);
    let mut sentinel_alt = 1_i32;
    while sentinel_hash_1024(sentinel_alt, chosen_multiplier) == home_slot_zero {
        sentinel_alt += 1;
    }

    let mut keys = [sentinel_zero; 1024];
    keys[home_slot_zero] = sentinel_alt;
    let mut counts = [0_u32; 1024];
    let mut used_bits = [0_u64; 16];
    let mut unique_count = 0_usize;

    for &key in &unique_probe[..unique_probe_len] {
        let slot = sentinel_hash_1024(key, multiplier_or_default(chosen_multiplier));
        let word_idx = slot >> 6;
        let mask = 1_u64 << (slot & 63);
        if (used_bits[word_idx] & mask) == 0 {
            used_bits[word_idx] |= mask;
            keys[slot] = key;
            unique_count += 1;
        }
    }

    for &item in slice.iter() {
        let val = item.to_key();
        let home_slot = sentinel_hash_1024(val, chosen_multiplier);
        if keys[home_slot] == val {
            counts[home_slot] += 1;
        } else {
            let mut slot = home_slot;
            loop {
                let word_idx = slot >> 6;
                let mask = 1_u64 << (slot & 63);
                if (used_bits[word_idx] & mask) == 0 {
                    if unique_count >= 384 {
                        return HashAggregationOutcome {
                            sorted: false,
                            span_exceeds_horizon,
                        };
                    }
                    used_bits[word_idx] |= mask;
                    keys[slot] = val;
                    counts[slot] = 1;
                    unique_count += 1;
                    break;
                }
                if keys[slot] == val {
                    counts[slot] += 1;
                    break;
                }
                slot = (slot + 1) & 1023;
            }
        }
    }

    if len <= u32::MAX as usize {
        let mut packed_entries = [0_u64; 384];
        let mut entry_count = 0_usize;
        for (word_idx, &word) in used_bits.iter().enumerate() {
            let mut bits = word;
            let base_slot = word_idx << 6;
            while bits != 0 {
                let slot = base_slot + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let c = counts[slot];
                if c > 0 {
                    let ordered_key = (keys[slot] as u32) ^ 0x8000_0000;
                    packed_entries[entry_count] = (u64::from(ordered_key) << 32) | u64::from(c);
                    entry_count += 1;
                }
            }
        }

        for i in 1..entry_count {
            let current = packed_entries[i];
            let mut j = i;
            while j > 0 && packed_entries[j - 1] > current {
                packed_entries[j] = packed_entries[j - 1];
                j -= 1;
            }
            packed_entries[j] = current;
        }

        let mut offset = 0_usize;
        for &packed in &packed_entries[..entry_count] {
            let key = (((packed >> 32) as u32) ^ 0x8000_0000) as i32;
            let count = (packed as u32) as usize;
            slice[offset..offset + count].fill(T::from_key(key));
            offset += count;
        }
    } else {
        let mut entries = [KeyCount32 { key: 0, count: 0 }; 384];
        let mut entry_count = 0_usize;
        for (word_idx, &word) in used_bits.iter().enumerate() {
            let mut bits = word;
            let base_slot = word_idx << 6;
            while bits != 0 {
                let slot = base_slot + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let c = counts[slot];
                if c > 0 {
                    entries[entry_count] = KeyCount32 {
                        key: keys[slot],
                        count: c as usize,
                    };
                    entry_count += 1;
                }
            }
        }

        for i in 1..entry_count {
            let current = entries[i];
            let mut j = i;
            while j > 0 && entries[j - 1].key > current.key {
                entries[j] = entries[j - 1];
                j -= 1;
            }
            entries[j] = current;
        }

        let mut offset = 0_usize;
        for entry in &entries[..entry_count] {
            slice[offset..offset + entry.count].fill(T::from_key(entry.key));
            offset += entry.count;
        }
    }

    HashAggregationOutcome {
        sorted: true,
        span_exceeds_horizon,
    }
}

#[inline(always)]
const fn multiplier_or_default(multiplier: u32) -> u32 {
    multiplier
}

/// Attempts to sort `slice` using a stack-allocated 1024-slot dual-sentinel hash table
/// when the input contains at most 384 distinct keys.
pub fn try_sentinel_hash_aggregation(slice: &mut [i32]) -> HashAggregationOutcome {
    try_sentinel_hash_aggregation_generic(slice)
}

/// Generic cache-bounded dense range counting sort for any `T: LatticeKey32`.
#[must_use]
pub(crate) fn try_dense_range_count_generic<T: LatticeKey32>(slice: &mut [T]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
        return false;
    }

    let first_key = slice[0].to_key();
    let mut min_val = first_key;
    let mut max_val = first_key;
    for &item in &slice[1..] {
        let x = item.to_key();
        if x < min_val {
            min_val = x;
        }
        if x > max_val {
            max_val = x;
        }
    }

    let span = (i64::from(max_val) - i64::from(min_val)) as u64;
    if span == 0 {
        return true;
    }

    let horizon = (len as u64).min(DENSE_RANGE_MAX_SPAN);
    if span > horizon {
        return false;
    }

    let table_len = (span + 1) as usize;
    let mut density = vec![0_u32; table_len];
    let base = i64::from(min_val);
    for &item in slice.iter() {
        let offset_idx = (i64::from(item.to_key()) - base) as usize;
        density[offset_idx] += 1;
    }

    let mut offset = 0_usize;
    let indexed_density = density.iter().enumerate();
    for (idx, &count) in indexed_density {
        if count > 0 {
            let val = T::from_key((base + idx as i64) as i32);
            let count_usize = count as usize;
            slice[offset..offset + count_usize].fill(val);
            offset += count_usize;
        }
    }
    true
}

/// Attempts to sort `slice` using offset counting sort when the value range `max - min`
/// fits within the cache-friendly span horizon.
#[must_use]
pub fn try_dense_range_count(slice: &mut [i32]) -> bool {
    try_dense_range_count_generic(slice)
}

/// Executes a single 8-bit radix scatter pass from `src` into `dst` using precomputed `histogram`.
#[inline]
pub(crate) fn scatter_pass_generic<T: LatticeKey32, const SHIFT: usize, const FLIP_SIGN: bool>(
    src: &[T],
    dst: &mut [T],
    histogram: &[u32; 256],
) {
    let mut offsets = [0_usize; 256];
    let mut sum = 0_usize;
    let indexed_histogram = histogram.iter().enumerate();
    for (i, &count) in indexed_histogram {
        offsets[i] = sum;
        sum += count as usize;
    }

    for &item in src {
        let raw = (item.to_key() as u32) >> SHIFT;
        let bucket = if FLIP_SIGN {
            ((raw ^ 0x80) & 0xff) as usize
        } else {
            (raw & 0xff) as usize
        };
        dst[offsets[bucket]] = item;
        offsets[bucket] += 1;
    }
}

/// Sorts an L1/L2-local sub-bucket on its lower 16 bits (Byte 0 and Byte 1).
fn l2_local_2pass_radix_sort<T: LatticeKey32>(slice_sub: &mut [T], scratch_sub: &mut [T]) {
    let len = slice_sub.len();
    if len <= SMALL_SORT_CUTOFF {
        if len > 1 {
            T::sort_small_in_place(slice_sub);
        }
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];

    let (chunks, remainder) = slice_sub.as_chunks::<4>();
    for chunk in chunks {
        let u0 = chunk[0].to_key() as u32;
        let u1 = chunk[1].to_key() as u32;
        let u2 = chunk[2].to_key() as u32;
        let u3 = chunk[3].to_key() as u32;

        h0[(u0 & 0xff) as usize] += 1;
        h1[((u0 >> 8) & 0xff) as usize] += 1;

        h0[(u1 & 0xff) as usize] += 1;
        h1[((u1 >> 8) & 0xff) as usize] += 1;

        h0[(u2 & 0xff) as usize] += 1;
        h1[((u2 >> 8) & 0xff) as usize] += 1;

        h0[(u3 & 0xff) as usize] += 1;
        h1[((u3 >> 8) & 0xff) as usize] += 1;
    }
    for &item in remainder {
        let u = item.to_key() as u32;
        h0[(u & 0xff) as usize] += 1;
        h1[((u >> 8) & 0xff) as usize] += 1;
    }

    let can_skip = len <= u32::MAX as usize;
    let n32 = len as u32;
    let mut in_scratch = false;

    if !(can_skip && h0.contains(&n32)) {
        scatter_pass_generic::<T, 0, false>(slice_sub, scratch_sub, &h0);
        in_scratch = true;
    }
    if !(can_skip && h1.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 8, false>(scratch_sub, slice_sub, &h1);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 8, false>(slice_sub, scratch_sub, &h1);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice_sub.copy_from_slice(scratch_sub);
    }
}

/// Sorts an L2-resident bucket on its lower 24 bits (3 passes of 8-bit LSD radix sort)
/// when the input initially resides in `scratch_bucket` and the final output must be in `slice_bucket`.
pub(crate) fn l2_local_3pass_radix_sort<T: LatticeKey32>(
    slice_bucket: &mut [T],
    scratch_bucket: &mut [T],
) {
    let len = scratch_bucket.len();
    if len <= SMALL_SORT_CUTOFF {
        slice_bucket.copy_from_slice(scratch_bucket);
        if len > 1 {
            T::sort_small_in_place(slice_bucket);
        }
        return;
    }

    let can_skip = len <= u32::MAX as usize;
    let n32 = len as u32;

    if len > 262_144 {
        let mut h2 = [0_u32; 256];
        let (chunks, remainder) = scratch_bucket.as_chunks::<4>();
        for chunk in chunks {
            h2[((chunk[0].to_key() as u32 >> 16) & 0xff) as usize] += 1;
            h2[((chunk[1].to_key() as u32 >> 16) & 0xff) as usize] += 1;
            h2[((chunk[2].to_key() as u32 >> 16) & 0xff) as usize] += 1;
            h2[((chunk[3].to_key() as u32 >> 16) & 0xff) as usize] += 1;
        }
        for &item in remainder {
            h2[((item.to_key() as u32 >> 16) & 0xff) as usize] += 1;
        }

        if can_skip && h2.contains(&n32) {
            slice_bucket.copy_from_slice(scratch_bucket);
            l2_local_2pass_radix_sort(slice_bucket, scratch_bucket);
            return;
        }

        let mut sub_starts = [0_usize; 256];
        let mut sum = 0_usize;
        let indexed_h2 = h2.iter().enumerate();
        for (i, &count) in indexed_h2 {
            sub_starts[i] = sum;
            sum += count as usize;
        }

        scatter_pass_generic::<T, 16, false>(scratch_bucket, slice_bucket, &h2);

        for b in 0..256 {
            let count = h2[b] as usize;
            if count <= 1 {
                continue;
            }
            let start = sub_starts[b];
            let end = start + count;
            l2_local_2pass_radix_sort(
                &mut slice_bucket[start..end],
                &mut scratch_bucket[start..end],
            );
        }
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];

    let (chunks, remainder) = scratch_bucket.as_chunks::<4>();
    for chunk in chunks {
        let u0 = chunk[0].to_key() as u32;
        let u1 = chunk[1].to_key() as u32;
        let u2 = chunk[2].to_key() as u32;
        let u3 = chunk[3].to_key() as u32;

        h0[(u0 & 0xff) as usize] += 1;
        h1[((u0 >> 8) & 0xff) as usize] += 1;
        h2[((u0 >> 16) & 0xff) as usize] += 1;

        h0[(u1 & 0xff) as usize] += 1;
        h1[((u1 >> 8) & 0xff) as usize] += 1;
        h2[((u1 >> 16) & 0xff) as usize] += 1;

        h0[(u2 & 0xff) as usize] += 1;
        h1[((u2 >> 8) & 0xff) as usize] += 1;
        h2[((u2 >> 16) & 0xff) as usize] += 1;

        h0[(u3 & 0xff) as usize] += 1;
        h1[((u3 >> 8) & 0xff) as usize] += 1;
        h2[((u3 >> 16) & 0xff) as usize] += 1;
    }
    for &item in remainder {
        let u = item.to_key() as u32;
        h0[(u & 0xff) as usize] += 1;
        h1[((u >> 8) & 0xff) as usize] += 1;
        h2[((u >> 16) & 0xff) as usize] += 1;
    }

    let mut in_scratch = true;

    if !(can_skip && h0.contains(&n32)) {
        scatter_pass_generic::<T, 0, false>(scratch_bucket, slice_bucket, &h0);
        in_scratch = false;
    }
    if !(can_skip && h1.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 8, false>(scratch_bucket, slice_bucket, &h1);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 8, false>(slice_bucket, scratch_bucket, &h1);
            in_scratch = true;
        }
    }
    if !(can_skip && h2.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 16, false>(scratch_bucket, slice_bucket, &h2);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 16, false>(slice_bucket, scratch_bucket, &h2);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice_bucket.copy_from_slice(scratch_bucket);
    }
}

/// Fills `h3` with the Most Significant Byte histogram of `slice` (sign-flipped onto the
/// unsigned order lattice).
pub(crate) fn msb_histogram_generic<T: LatticeKey32>(slice: &[T], h3: &mut [u32; 256]) {
    let (chunks, remainder) = slice.as_chunks::<4>();
    for chunk in chunks {
        h3[(((chunk[0].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((chunk[1].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((chunk[2].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((chunk[3].to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
    }
    for &item in remainder {
        h3[(((item.to_key() as u32) >> 24) ^ 0x80) as usize] += 1;
    }
}

/// Reports whether the 8-point strided envelope of `slice` is monotone (at most one direction),
/// which identifies barely perturbed sorted data that the LSD cascade scatters while preserving.
fn has_monotone_envelope_generic<T: LatticeKey32>(slice: &[T]) -> bool {
    let step = slice.len() / 8;
    let mut prev = 0_i8;
    for i in 1..8 {
        let a = slice[(i - 1) * step].to_key();
        let b = slice[i * step].to_key();
        let dir = if b > a {
            1_i8
        } else if b < a {
            -1_i8
        } else {
            0
        };
        if dir != 0 {
            if prev != 0 && dir != prev {
                return false;
            }
            prev = dir;
        }
    }
    true
}

/// Hybrid MSB-first + L2-local 3-pass LSD radix sort for large slices ($N \ge 1{,}500{,}000$).
pub(crate) fn radix_msb_l2_hybrid_sort_generic<T: LatticeKey32>(slice: &mut [T]) {
    let mut h3 = [0_u32; 256];
    msb_histogram_generic(slice, &mut h3);
    radix_msb_from_histogram_generic(slice, &h3);
}

/// MSB-first partition into 256 L2-resident buckets followed by a local 3-pass LSD sort per
/// bucket, driven by an already computed Most Significant Byte `h3`.
pub(crate) fn radix_msb_from_histogram_generic<T: LatticeKey32>(slice: &mut [T], h3: &[u32; 256]) {
    let len = slice.len();
    let h3 = *h3;
    let mut scratch = vec![T::ZERO; len];
    let can_skip = len <= u32::MAX as usize;
    let n32 = len as u32;
    if can_skip && h3.contains(&n32) {
        scratch.copy_from_slice(slice);
        l2_local_3pass_radix_sort(slice, &mut scratch);
        return;
    }

    let mut bucket_starts = [0_usize; 256];
    let mut sum = 0_usize;
    let indexed_h3 = h3.iter().enumerate();
    for (i, &count) in indexed_h3 {
        bucket_starts[i] = sum;
        sum += count as usize;
    }

    scatter_pass_generic::<T, 24, true>(slice, &mut scratch, &h3);

    for bucket in 0..256 {
        let count = h3[bucket] as usize;
        if count == 0 {
            continue;
        }
        let start = bucket_starts[bucket];
        let end = start + count;
        if count == 1 {
            slice[start] = scratch[start];
        } else if count <= SMALL_SORT_CUTOFF {
            slice[start..end].copy_from_slice(&scratch[start..end]);
            T::sort_small_in_place(&mut slice[start..end]);
        } else {
            l2_local_3pass_radix_sort(&mut slice[start..end], &mut scratch[start..end]);
        }
    }
}

/// Runs the 4-pass fused 8-bit LSD radix stage on `slice` without running the adaptive heuristics.
pub fn radix_lsd_fused_sort(slice: &mut [i32]) {
    radix_lsd_fused_sort_generic(slice);
}

/// Runs the MSB-first + L2-local radix cascade on `slice` without running the adaptive heuristics.
pub fn radix_msb_l2_hybrid_sort(slice: &mut [i32]) {
    radix_msb_l2_hybrid_sort_generic(slice);
}

/// 4-pass 8-bit LSD radix sort with a fused single-pass 4x256 histogram and degenerate-byte skipping.
pub(crate) fn radix_lsd_fused_sort_generic<T: LatticeKey32>(slice: &mut [T]) {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
        T::sort_small_in_place(slice);
        return;
    }
    if len >= MSB_L2_PARTITION_THRESHOLD {
        radix_msb_l2_hybrid_sort_generic(slice);
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];
    let mut h3 = [0_u32; 256];

    let (chunks, remainder) = slice.as_chunks::<4>();
    for chunk in chunks {
        let u0 = chunk[0].to_key() as u32;
        let u1 = chunk[1].to_key() as u32;
        let u2 = chunk[2].to_key() as u32;
        let u3 = chunk[3].to_key() as u32;

        h0[(u0 & 0xff) as usize] += 1;
        h1[((u0 >> 8) & 0xff) as usize] += 1;
        h2[((u0 >> 16) & 0xff) as usize] += 1;
        h3[((u0 >> 24) ^ 0x80) as usize] += 1;

        h0[(u1 & 0xff) as usize] += 1;
        h1[((u1 >> 8) & 0xff) as usize] += 1;
        h2[((u1 >> 16) & 0xff) as usize] += 1;
        h3[((u1 >> 24) ^ 0x80) as usize] += 1;

        h0[(u2 & 0xff) as usize] += 1;
        h1[((u2 >> 8) & 0xff) as usize] += 1;
        h2[((u2 >> 16) & 0xff) as usize] += 1;
        h3[((u2 >> 24) ^ 0x80) as usize] += 1;

        h0[(u3 & 0xff) as usize] += 1;
        h1[((u3 >> 8) & 0xff) as usize] += 1;
        h2[((u3 >> 16) & 0xff) as usize] += 1;
        h3[((u3 >> 24) ^ 0x80) as usize] += 1;
    }
    for &item in remainder {
        let u = item.to_key() as u32;
        h0[(u & 0xff) as usize] += 1;
        h1[((u >> 8) & 0xff) as usize] += 1;
        h2[((u >> 16) & 0xff) as usize] += 1;
        h3[((u >> 24) ^ 0x80) as usize] += 1;
    }

    let can_skip = len <= u32::MAX as usize;
    let n32 = len as u32;

    // Entropy gate: an MSB-first cascade only repays its extra partition when every byte is
    // populated (no degenerate skip available), the top byte is spread over the whole fan-out,
    // and the envelope is not monotone (near-sorted data keeps its locality under LSD).
    if len >= MSB_ENTROPY_GATE_MIN_LEN
        && ![&h0, &h1, &h2, &h3].iter().any(|hist| hist.contains(&n32))
    {
        let occupied = h3.iter().filter(|&&count| count > 0).count();
        let widest = h3.iter().copied().max().unwrap_or(0);
        if occupied > 200 && widest.saturating_mul(8) < n32 && !has_monotone_envelope_generic(slice)
        {
            if false && std::env::var_os("LATT_DEBUG").is_some() {
                eprintln!("MSB gate fired len={len} occupied={occupied} widest={widest}");
            }
            radix_msb_from_histogram_generic(slice, &h3);
            return;
        }
    }

    let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX];
    let mut heap_buf: Vec<T>;
    let scratch: &mut [T] = if len <= STACK_SCRATCH_MAX {
        &mut stack_buf[..len]
    } else {
        heap_buf = vec![T::ZERO; len];
        &mut heap_buf
    };
    let mut in_scratch = false;

    if !(can_skip && h0.contains(&n32)) {
        scatter_pass_generic::<T, 0, false>(slice, scratch, &h0);
        in_scratch = true;
    }
    if !(can_skip && h1.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 8, false>(scratch, slice, &h1);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 8, false>(slice, scratch, &h1);
            in_scratch = true;
        }
    }
    if !(can_skip && h2.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 16, false>(scratch, slice, &h2);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 16, false>(slice, scratch, &h2);
            in_scratch = true;
        }
    }
    if !(can_skip && h3.contains(&n32)) {
        if in_scratch {
            scatter_pass_generic::<T, 24, true>(scratch, slice, &h3);
            in_scratch = false;
        } else {
            scatter_pass_generic::<T, 24, true>(slice, scratch, &h3);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice.copy_from_slice(scratch);
    }
}

/// Generic 32-bit adaptive sorting pipeline for any `T: LatticeKey32` (`i32`, `u32`, `f32`).
pub fn lattice_sort_generic_32<T: LatticeKey32>(data: &mut [T]) {
    if data.len() <= SMALL_SORT_CUTOFF {
        if data.len() > 1 {
            T::sort_small_in_place(data);
        }
        return;
    }
    if try_bitonic_run_merge_generic(data) {
        return;
    }
    let outcome = try_sentinel_hash_aggregation_generic(data);
    if outcome.sorted {
        return;
    }
    if !outcome.span_exceeds_horizon && try_dense_range_count_generic(data) {
        return;
    }
    radix_lsd_fused_sort_generic(data);
}

/// Sorts a mutable slice of 32-bit signed integers (`i32`) in ascending order using the
/// adaptive `latticesort` pipeline.
#[inline]
pub fn lattice_sort_i32(data: &mut [i32]) {
    lattice_sort_generic_32(data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage1_bitonic_and_plateau_snap_back() {
        for &len in &[1000_usize, 3000_usize] {
            let mut asc: Vec<i32> = (0..len as i32).collect();
            assert!(try_bitonic_run_merge(&mut asc));
            assert!(asc.is_sorted());

            let mut desc: Vec<i32> = (0..len as i32).rev().collect();
            let mut desc_expected = desc.clone();
            desc_expected.sort_unstable();
            assert!(try_bitonic_run_merge(&mut desc));
            assert_eq!(desc, desc_expected);

            let mid = len / 2;
            let mut peak: Vec<i32> = (0..mid as i32)
                .chain((0..(len - mid) as i32).rev())
                .collect();
            let mut peak_expected = peak.clone();
            peak_expected.sort_unstable();
            assert!(try_bitonic_run_merge(&mut peak));
            assert_eq!(peak, peak_expected);

            let mut valley: Vec<i32> = (0..mid as i32).rev().chain(0..(len - mid) as i32).collect();
            let mut valley_expected = valley.clone();
            valley_expected.sort_unstable();
            assert!(try_bitonic_run_merge(&mut valley));
            assert_eq!(valley, valley_expected);
        }

        // Plateau snap-back branch: peak >= (len * 3) / 4 with constant prefix matching slice[peak / 2],
        // followed by a short ascending ramp and an unsorted tail containing values below and above 50.
        let n = 2000_usize;
        let mut plateau = Vec::with_capacity(n);
        plateau.extend(std::iter::repeat_n(50_i32, 1200));
        plateau.extend(51_i32..501_i32); // indices 1200..1650 -> peak = 1649 >= 1500
        for i in 1650..n {
            let v = if i % 2 == 0 {
                -(i as i32)
            } else {
                (i as i32) * 2
            };
            plateau.push(v);
        }
        let mut expected = plateau.clone();
        expected.sort_unstable();
        assert!(try_bitonic_run_merge(&mut plateau));
        assert_eq!(plateau, expected);
    }

    #[test]
    fn test_stage2_sentinel_hash_boundaries_and_home_slot_zero_collision() {
        // Home-slot-zero triple collision under SENTINEL_HASH_MULTIPLIERS_32[0].
        let mult0 = SENTINEL_HASH_MULTIPLIERS_32[0];
        let home0 = sentinel_hash_1024(0, mult0);
        let mut colliding_key = 2_i32;
        while sentinel_hash_1024(colliding_key, mult0) != home0 {
            colliding_key += 1;
        }
        let mut triple = vec![0_i32; 3000];
        for (idx, slot) in triple.iter_mut().enumerate() {
            *slot = if idx % 3 == 0 { 0 } else { 1 };
        }
        // Place colliding_key outside the probe windows so mult0 remains chosen, forcing linear probing from home0.
        triple[1] = colliding_key;
        triple[2] = colliding_key;
        let mut triple_expected = triple.clone();
        triple_expected.sort_unstable();
        let outcome_triple = try_sentinel_hash_aggregation(&mut triple);
        assert!(outcome_triple.sorted);
        assert_eq!(triple, triple_expected);

        // Exact cardinality boundary: K = 384 (succeeds) vs K = 385 (aborts cleanly).
        let n = 4096_usize;
        let base_prefix = (n * 85) / 100;
        let mut k384 = Vec::with_capacity(n);
        for i in 0..base_prefix {
            k384.push((i % 10) as i32 * 1_000_000);
        }
        for i in 0..(n - base_prefix) {
            let distinct_idx = if i < 374 { 10 + i } else { 10 };
            k384.push((distinct_idx as i32) * 1_000_000);
        }
        let mut k384_expected = k384.clone();
        k384_expected.sort_unstable();
        let outcome_384 = try_sentinel_hash_aggregation(&mut k384);
        assert!(outcome_384.sorted);
        assert_eq!(k384, k384_expected);

        let mut k385 = Vec::with_capacity(n);
        for i in 0..base_prefix {
            k385.push((i % 10) as i32 * 1_000_000);
        }
        for i in 0..(n - base_prefix) {
            let distinct_idx = if i < 375 { 10 + i } else { 10 };
            k385.push((distinct_idx as i32) * 1_000_000);
        }
        let outcome_385 = try_sentinel_hash_aggregation(&mut k385);
        assert!(!outcome_385.sorted);

        // IEEE-754 -0.0_f32 and +0.0_f32 separation inside the dual-sentinel L1 hash table.
        let mut f32_signed_zeros = Vec::with_capacity(3000);
        for i in 0..3000 {
            match i % 4 {
                0 => f32_signed_zeros.push(0.0_f32),
                1 => f32_signed_zeros.push(-0.0_f32),
                2 => f32_signed_zeros.push(1.5_f32),
                _ => f32_signed_zeros.push(-1.5_f32),
            }
        }
        let mut f32_expected = f32_signed_zeros.clone();
        f32_expected.sort_by(f32::total_cmp);
        let f32_outcome = try_sentinel_hash_aggregation_generic(&mut f32_signed_zeros);
        assert!(f32_outcome.sorted);
        for (a, b) in f32_signed_zeros.iter().zip(f32_expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_stage3_dense_range_count_direct() {
        let mut equal = vec![7_i32; 1000];
        assert!(try_dense_range_count(&mut equal));

        let n = 2000_usize;
        let mut exact_horizon: Vec<i32> = (0..n).map(|i| ((i * 7) % (n + 1)) as i32).collect();
        exact_horizon[0] = 0;
        exact_horizon[1] = n as i32; // span == len
        let mut expected = exact_horizon.clone();
        expected.sort_unstable();
        assert!(try_dense_range_count(&mut exact_horizon));
        assert_eq!(exact_horizon, expected);

        let mut exceeds_horizon: Vec<i32> = (0..n).map(|i| (i % 50) as i32).collect();
        exceeds_horizon[0] = 0;
        exceeds_horizon[1] = (n + 1) as i32; // span == len + 1
        assert!(!try_dense_range_count(&mut exceeds_horizon));
    }

    #[test]
    fn test_stage4_l2_sub_bucket_split_above_262k() {
        let n = 270_000_usize; // > 262_144 threshold in l2_local_3pass_radix_sort

        // Case A: variable Byte 2
        let mut seed = 0x1357_2468_u32;
        let mut scratch_a = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            scratch_a.push((0x2A00_0000_u32 | (seed & 0x00ff_ffff)) as i32);
        }
        let mut expected_a = scratch_a.clone();
        expected_a.sort_unstable();
        let mut slice_a = vec![0_i32; n];
        l2_local_3pass_radix_sort(&mut slice_a, &mut scratch_a);
        assert_eq!(slice_a, expected_a);

        // Case B: constant Byte 2 (h2.contains(&n32) fast path)
        let mut scratch_b = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            scratch_b.push((0x2A55_0000_u32 | (seed & 0x0000_ffff)) as i32);
        }
        let mut expected_b = scratch_b.clone();
        expected_b.sort_unstable();
        let mut slice_b = vec![0_i32; n];
        l2_local_3pass_radix_sort(&mut slice_b, &mut scratch_b);
        assert_eq!(slice_b, expected_b);
    }
}
