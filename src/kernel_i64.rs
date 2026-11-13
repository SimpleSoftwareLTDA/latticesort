//! 64-bit adaptive sorting kernel (`lattice_sort_i64` and `lattice_sort_generic_64`).
//!
//! Combines all four adaptive stages for 64-bit bijective types (`i64`, `u64`, `f64`):
//! 1. `try_bitonic_run_merge_64`: linear-time $O(n)$ run detection and merge.
//! 2. `try_sentinel_hash_aggregation_64`: 1024-slot dual-sentinel L1 hash table ($K \le 384$).
//! 3. `try_dense_range_count_64`: cache-bounded offset counting sort when $E_{\max} - E_{\min}$ fits in cache.
//! 4. 8-pass 8-bit LSD radix sort with 4x unrolled single-pass histograms, stack buffer for
//!    $N \le 1024$, and compile-time `const SHIFT` / `const FLIP_SIGN` scatter passes.

use crate::candidate::HashAggregationOutcome;

const DENSE_RANGE_MAX_SPAN_64: u128 = 1_048_576;
pub(crate) const SMALL_SORT_CUTOFF_64: usize = 384;
const STACK_SCRATCH_MAX_64: usize = 1024;

const SENTINEL_HASH_MULTIPLIERS_64: [u64; 4] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
    0xd6e8_feb8_6659_fd93,
];

/// Bijective 64-bit key mapping onto the signed `i64` total-order lattice.
pub trait LatticeKey64: Copy + Send + Sync {
    /// Zero-initialized default value used for scratch buffers.
    const ZERO: Self;

    /// Projects `self` onto `i64` preserving total order ($a \le b \iff \text{to\_key}(a) \le \text{to\_key}(b)$).
    fn to_key(self) -> i64;

    /// Inverse bijection reconstructing `Self` from its `i64` lattice key.
    fn from_key(key: i64) -> Self;

    /// Sorts a small slice ($\le 384$ elements) in-place without heap allocation.
    fn sort_small_in_place(slice: &mut [Self]);
}

impl LatticeKey64 for i64 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn to_key(self) -> i64 {
        self
    }

    #[inline(always)]
    fn from_key(key: i64) -> Self {
        key
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable();
    }
}

impl LatticeKey64 for u64 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn to_key(self) -> i64 {
        (self ^ 0x8000_0000_0000_0000) as i64
    }

    #[inline(always)]
    fn from_key(key: i64) -> Self {
        (key as u64) ^ 0x8000_0000_0000_0000
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable();
    }
}

impl LatticeKey64 for f64 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn to_key(self) -> i64 {
        let bits = self.to_bits();
        let mask = (((bits as i64) >> 63) as u64) & 0x7fff_ffff_ffff_ffff;
        (bits ^ mask) as i64
    }

    #[inline(always)]
    fn from_key(key: i64) -> Self {
        let bits = key as u64;
        let mask = ((key >> 63) as u64) & 0x7fff_ffff_ffff_ffff;
        Self::from_bits(bits ^ mask)
    }

    #[inline]
    fn sort_small_in_place(slice: &mut [Self]) {
        slice.sort_unstable_by(Self::total_cmp);
    }
}

#[derive(Clone, Copy)]
struct KeyCount64 {
    key: i64,
    count: usize,
}

fn insertion_sort_i64(slice: &mut [i64]) {
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

fn merge_sorted_halves_64<T: LatticeKey64>(slice: &mut [T], split: usize) {
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
    let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX_64];
    let mut heap_buf: Vec<T>;
    let merged: &mut [T] = if len <= STACK_SCRATCH_MAX_64 {
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

#[must_use]
pub(crate) fn try_bitonic_run_merge_64<T: LatticeKey64>(slice: &mut [T]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF_64 {
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
            merge_sorted_halves_64(slice, valley + 1);
            return true;
        }
        return false;
    }

    let mut tail = peak + 1;
    while tail + 1 < len && slice[tail].to_key() >= slice[tail + 1].to_key() {
        tail += 1;
    }
    if tail == len - 1 {
        let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX_64];
        let mut heap_buf: Vec<T>;
        let merged: &mut [T] = if len <= STACK_SCRATCH_MAX_64 {
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
        lattice_sort_generic_64(&mut slice[peak + 1..]);
        merge_sorted_halves_64(slice, peak + 1);
        return true;
    }

    false
}

#[inline]
fn sentinel_hash_64(val: i64, multiplier: u64) -> usize {
    let u = val as u64;
    (((u ^ (u >> 30)).wrapping_mul(multiplier)) >> 54) as usize
}

pub(crate) fn try_sentinel_hash_aggregation_64<T: LatticeKey64>(
    slice: &mut [T],
) -> HashAggregationOutcome {
    let len = slice.len();
    if len <= STACK_SCRATCH_MAX_64 {
        return HashAggregationOutcome {
            sorted: false,
            span_exceeds_horizon: true,
        };
    }

    let mut probe = [0_i64; 128];
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
    insertion_sort_i64(&mut probe);

    let probe_span = (i128::from(probe[127]) - i128::from(probe[0])) as u128;
    let horizon = (len as u128).min(DENSE_RANGE_MAX_SPAN_64);
    let span_exceeds_horizon = probe_span > horizon;

    let mut unique_probe = [0_i64; 128];
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

    let mut chosen_multiplier = SENTINEL_HASH_MULTIPLIERS_64[0];
    for &multiplier in &SENTINEL_HASH_MULTIPLIERS_64 {
        let mut occupied_bits = [0_u64; 16];
        let mut collision_free = true;
        for &key in &unique_probe[..unique_probe_len] {
            let slot = sentinel_hash_64(key, multiplier);
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

    let sentinel_zero = 0_i64;
    let home_slot_zero = sentinel_hash_64(sentinel_zero, chosen_multiplier);
    let mut sentinel_alt = 1_i64;
    while sentinel_hash_64(sentinel_alt, chosen_multiplier) == home_slot_zero {
        sentinel_alt += 1;
    }

    let mut keys = [sentinel_zero; 1024];
    keys[home_slot_zero] = sentinel_alt;
    let mut counts = [0_u32; 1024];
    let mut used_bits = [0_u64; 16];
    let mut unique_count = 0_usize;

    for &key in &unique_probe[..unique_probe_len] {
        let slot = sentinel_hash_64(key, chosen_multiplier);
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
        let home_slot = sentinel_hash_64(val, chosen_multiplier);
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

    let mut entries = [KeyCount64 { key: 0, count: 0 }; 384];
    let mut entry_count = 0_usize;
    for (word_idx, &word) in used_bits.iter().enumerate() {
        let mut bits = word;
        let base_slot = word_idx << 6;
        while bits != 0 {
            let slot = base_slot + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let c = counts[slot];
            if c > 0 {
                entries[entry_count] = KeyCount64 {
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
    HashAggregationOutcome {
        sorted: true,
        span_exceeds_horizon,
    }
}

#[must_use]
pub(crate) fn try_dense_range_count_64<T: LatticeKey64>(slice: &mut [T]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF_64 {
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

    let span = (i128::from(max_val) - i128::from(min_val)) as u128;
    if span == 0 {
        return true;
    }

    let horizon = (len as u128).min(DENSE_RANGE_MAX_SPAN_64);
    if span > horizon {
        return false;
    }

    let table_len = (span + 1) as usize;
    let mut density = vec![0_u32; table_len];
    let base = i128::from(min_val);
    for &item in slice.iter() {
        let offset_idx = (i128::from(item.to_key()) - base) as usize;
        density[offset_idx] += 1;
    }

    let mut offset = 0_usize;
    let indexed_density = density.iter().enumerate();
    for (idx, &count) in indexed_density {
        if count > 0 {
            let val = T::from_key((base + idx as i128) as i64);
            let count_usize = count as usize;
            slice[offset..offset + count_usize].fill(val);
            offset += count_usize;
        }
    }
    true
}

#[inline]
pub(crate) fn scatter_pass_64<T: LatticeKey64, const SHIFT: usize, const FLIP_SIGN: bool>(
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
        let raw = (item.to_key() as u64) >> SHIFT;
        let bucket = if FLIP_SIGN {
            ((raw ^ 0x80) & 0xff) as usize
        } else {
            (raw & 0xff) as usize
        };
        dst[offsets[bucket]] = item;
        offsets[bucket] += 1;
    }
}

/// 8-pass 8-bit LSD radix sort with 4x unrolled single-pass histograms and degenerate-byte skipping.
pub(crate) fn radix_lsd_fused_sort_64<T: LatticeKey64>(slice: &mut [T]) {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF_64 {
        T::sort_small_in_place(slice);
        return;
    }

    let mut histograms = [[0_u32; 256]; 8];
    let (chunks, remainder) = slice.as_chunks::<4>();
    for chunk in chunks {
        let u0 = chunk[0].to_key() as u64;
        let u1 = chunk[1].to_key() as u64;
        let u2 = chunk[2].to_key() as u64;
        let u3 = chunk[3].to_key() as u64;

        for u in [u0, u1, u2, u3] {
            histograms[0][(u & 0xff) as usize] += 1;
            histograms[1][((u >> 8) & 0xff) as usize] += 1;
            histograms[2][((u >> 16) & 0xff) as usize] += 1;
            histograms[3][((u >> 24) & 0xff) as usize] += 1;
            histograms[4][((u >> 32) & 0xff) as usize] += 1;
            histograms[5][((u >> 40) & 0xff) as usize] += 1;
            histograms[6][((u >> 48) & 0xff) as usize] += 1;
            histograms[7][((u >> 56) ^ 0x80) as usize] += 1;
        }
    }
    for &item in remainder {
        let u = item.to_key() as u64;
        histograms[0][(u & 0xff) as usize] += 1;
        histograms[1][((u >> 8) & 0xff) as usize] += 1;
        histograms[2][((u >> 16) & 0xff) as usize] += 1;
        histograms[3][((u >> 24) & 0xff) as usize] += 1;
        histograms[4][((u >> 32) & 0xff) as usize] += 1;
        histograms[5][((u >> 40) & 0xff) as usize] += 1;
        histograms[6][((u >> 48) & 0xff) as usize] += 1;
        histograms[7][((u >> 56) ^ 0x80) as usize] += 1;
    }

    let can_skip = len <= u32::MAX as usize;
    let n32 = len as u32;
    let mut stack_buf = [T::ZERO; STACK_SCRATCH_MAX_64];
    let mut heap_buf: Vec<T>;
    let scratch: &mut [T] = if len <= STACK_SCRATCH_MAX_64 {
        &mut stack_buf[..len]
    } else {
        heap_buf = vec![T::ZERO; len];
        &mut heap_buf
    };
    let mut in_scratch = false;

    macro_rules! run_pass {
        ($idx:expr, $shift:expr, $flip:expr) => {
            if !(can_skip && histograms[$idx].contains(&n32)) {
                if in_scratch {
                    scatter_pass_64::<T, $shift, $flip>(scratch, slice, &histograms[$idx]);
                    in_scratch = false;
                } else {
                    scatter_pass_64::<T, $shift, $flip>(slice, scratch, &histograms[$idx]);
                    in_scratch = true;
                }
            }
        };
    }

    run_pass!(0, 0, false);
    run_pass!(1, 8, false);
    run_pass!(2, 16, false);
    run_pass!(3, 24, false);
    run_pass!(4, 32, false);
    run_pass!(5, 40, false);
    run_pass!(6, 48, false);
    run_pass!(7, 56, true);

    if in_scratch {
        slice.copy_from_slice(scratch);
    }
}

/// Generic 64-bit adaptive sorting pipeline for any `T: LatticeKey64` (`i64`, `u64`, `f64`).
pub fn lattice_sort_generic_64<T: LatticeKey64>(slice: &mut [T]) {
    if slice.len() <= SMALL_SORT_CUTOFF_64 {
        if slice.len() > 1 {
            T::sort_small_in_place(slice);
        }
        return;
    }
    if try_bitonic_run_merge_64(slice) {
        return;
    }
    let outcome = try_sentinel_hash_aggregation_64(slice);
    if outcome.sorted {
        return;
    }
    if !outcome.span_exceeds_horizon && try_dense_range_count_64(slice) {
        return;
    }
    radix_lsd_fused_sort_64(slice);
}

/// Sorts a mutable slice of 64-bit signed integers (`i64`) in ascending order using the
/// adaptive 64-bit `latticesort` pipeline.
#[inline]
pub fn lattice_sort_i64(slice: &mut [i64]) {
    lattice_sort_generic_64(slice);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_64bit_stage1_bitonic_and_plateau_snap_back() {
        let mid = 1000_usize;
        let len = 2000_usize;

        let mut peak: Vec<i64> = (0..mid as i64)
            .chain((0..(len - mid) as i64).rev())
            .collect();
        let mut peak_expected = peak.clone();
        peak_expected.sort_unstable();
        assert!(try_bitonic_run_merge_64(&mut peak));
        assert_eq!(peak, peak_expected);

        let mut valley: Vec<i64> = (0..mid as i64).rev().chain(0..(len - mid) as i64).collect();
        let mut valley_expected = valley.clone();
        valley_expected.sort_unstable();
        assert!(try_bitonic_run_merge_64(&mut valley));
        assert_eq!(valley, valley_expected);

        // Plateau snap-back branch
        let mut plateau = Vec::with_capacity(len);
        plateau.extend(std::iter::repeat_n(50_i64, 1200));
        plateau.extend(51_i64..501_i64);
        for i in 1650..len {
            let v = if i % 2 == 0 {
                -(i as i64)
            } else {
                (i as i64) * 2
            };
            plateau.push(v);
        }
        let mut plateau_expected = plateau.clone();
        plateau_expected.sort_unstable();
        assert!(try_bitonic_run_merge_64(&mut plateau));
        assert_eq!(plateau, plateau_expected);
    }

    #[test]
    fn test_64bit_stage2_sentinel_hash_and_signed_zero_f64() {
        let mult0 = SENTINEL_HASH_MULTIPLIERS_64[0];
        let home0 = sentinel_hash_64(0, mult0);
        let mut colliding_key = 2_i64;
        while sentinel_hash_64(colliding_key, mult0) != home0 {
            colliding_key += 1;
        }
        let mut triple = vec![0_i64; 3000];
        for (idx, slot) in triple.iter_mut().enumerate() {
            *slot = if idx % 3 == 0 { 0 } else { 1 };
        }
        triple[1] = colliding_key;
        triple[2] = colliding_key;
        let mut triple_expected = triple.clone();
        triple_expected.sort_unstable();
        let outcome = try_sentinel_hash_aggregation_64(&mut triple);
        assert!(outcome.sorted);
        assert_eq!(triple, triple_expected);

        let mut f64_signed_zeros = Vec::with_capacity(3000);
        for i in 0..3000 {
            match i % 4 {
                0 => f64_signed_zeros.push(0.0_f64),
                1 => f64_signed_zeros.push(-0.0_f64),
                2 => f64_signed_zeros.push(2.5_f64),
                _ => f64_signed_zeros.push(-2.5_f64),
            }
        }
        let mut f64_expected = f64_signed_zeros.clone();
        f64_expected.sort_by(f64::total_cmp);
        let f64_outcome = try_sentinel_hash_aggregation_64(&mut f64_signed_zeros);
        assert!(f64_outcome.sorted);
        for (a, b) in f64_signed_zeros.iter().zip(f64_expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_64bit_stage3_dense_range_count_extremes() {
        let n = 1000_usize;
        let mut near_max: Vec<i64> = (0..n).map(|i| i64::MAX - ((i * 13) % 500) as i64).collect();
        let mut near_max_expected = near_max.clone();
        near_max_expected.sort_unstable();
        assert!(try_dense_range_count_64(&mut near_max));
        assert_eq!(near_max, near_max_expected);

        let mut near_min: Vec<i64> = (0..n).map(|i| i64::MIN + ((i * 17) % 500) as i64).collect();
        let mut near_min_expected = near_min.clone();
        near_min_expected.sort_unstable();
        assert!(try_dense_range_count_64(&mut near_min));
        assert_eq!(near_min, near_min_expected);
    }

    #[test]
    fn test_64bit_stage4_odd_and_even_active_radix_passes() {
        let n = 2048_usize;
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;

        // 1 active byte (Byte 0 only -> in_scratch == true copy-back)
        let mut one_byte = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            one_byte.push((seed & 0xff) as i64);
        }
        let mut one_expected = one_byte.clone();
        one_expected.sort_unstable();
        radix_lsd_fused_sort_64(&mut one_byte);
        assert_eq!(one_byte, one_expected);

        // 3 active bytes (Bytes 0, 3, 7 -> in_scratch == true copy-back)
        let mut three_bytes = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let val = (seed & 0xff) | ((seed & 0xff00) << 16) | ((seed & 0x00ff_0000) << 40);
            three_bytes.push(val as i64);
        }
        let mut three_expected = three_bytes.clone();
        three_expected.sort_unstable();
        radix_lsd_fused_sort_64(&mut three_bytes);
        assert_eq!(three_bytes, three_expected);

        // 2 active bytes (Bytes 0, 7 -> in_scratch == false)
        let mut two_bytes = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let val = (seed & 0xff) | ((seed & 0xff00) << 48);
            two_bytes.push(val as i64);
        }
        let mut two_expected = two_bytes.clone();
        two_expected.sort_unstable();
        radix_lsd_fused_sort_64(&mut two_bytes);
        assert_eq!(two_bytes, two_expected);
    }
}
