//! 64-bit adaptive sorting kernel (`lattice_sort_i64`).
//!
//! Combines a 1024-slot dual-sentinel L1 hash table for low-cardinality 64-bit keys with
//! an 8-pass 8-bit LSD radix sort that computes all eight byte histograms in a single pass
//! and skips constant (zero-entropy) bytes.

const SENTINEL_HASH_MULTIPLIERS_64: [u64; 4] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
    0xd6e8_feb8_6659_fd93,
];

#[derive(Clone, Copy)]
struct KeyCount64 {
    key: i64,
    count: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SignBitMode {
    Preserve,
    FlipMostSignificantBit,
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

#[inline]
fn sentinel_hash_64(val: i64, multiplier: u64) -> usize {
    let u = val as u64;
    (((u ^ (u >> 30)).wrapping_mul(multiplier)) >> 54) as usize
}

fn try_sentinel_hash_aggregation_i64(slice: &mut [i64]) -> bool {
    let len = slice.len();
    if len <= 2048 {
        return false;
    }

    let mut probe = [0_i64; 128];
    let anchors = [len / 10, (len * 3) / 10, len / 2, (len * 7) / 10, len - 16];
    let indexed_anchors = anchors.iter().enumerate();
    for (window_idx, &base) in indexed_anchors {
        let dst_start = window_idx * 16;
        probe[dst_start..dst_start + 16].copy_from_slice(&slice[base..base + 16]);
    }

    let stride = ((len as u64 * 2_654_435_761) >> 32).max(1) as usize | 1;
    let mut idx = len / 7;
    for slot in &mut probe[80..] {
        *slot = slice[idx % len];
        idx = idx.wrapping_add(stride);
    }
    insertion_sort_i64(&mut probe);

    let mut unique_probe = [0_i64; 128];
    unique_probe[0] = probe[0];
    let mut unique_probe_len = 1_usize;
    for i in 1..128 {
        if probe[i] != probe[i - 1] {
            if unique_probe_len >= 80 {
                return false;
            }
            unique_probe[unique_probe_len] = probe[i];
            unique_probe_len += 1;
        }
    }

    let mut chosen_multiplier = SENTINEL_HASH_MULTIPLIERS_64[0];
    for &multiplier in &SENTINEL_HASH_MULTIPLIERS_64 {
        let mut occupied = [false; 1024];
        let mut collision_free = true;
        for &key in &unique_probe[..unique_probe_len] {
            let slot = sentinel_hash_64(key, multiplier);
            if occupied[slot] {
                collision_free = false;
                break;
            }
            occupied[slot] = true;
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
    let mut used = [false; 1024];
    let mut unique_count = 0_usize;

    for &key in &unique_probe[..unique_probe_len] {
        let slot = sentinel_hash_64(key, chosen_multiplier);
        if !used[slot] {
            used[slot] = true;
            keys[slot] = key;
            unique_count += 1;
        }
    }

    for &val in slice.iter() {
        let home_slot = sentinel_hash_64(val, chosen_multiplier);
        if keys[home_slot] == val {
            counts[home_slot] += 1;
        } else {
            let mut slot = home_slot;
            loop {
                if !used[slot] {
                    if unique_count >= 384 {
                        return false;
                    }
                    used[slot] = true;
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
    for slot in 0..1024 {
        if used[slot] && counts[slot] > 0 {
            entries[entry_count] = KeyCount64 {
                key: keys[slot],
                count: counts[slot] as usize,
            };
            entry_count += 1;
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
        slice[offset..offset + entry.count].fill(entry.key);
        offset += entry.count;
    }
    true
}

#[inline]
fn scatter_pass_i64(
    src: &[i64],
    dst: &mut [i64],
    histogram: &[u32; 256],
    shift: usize,
    sign_mode: SignBitMode,
) {
    let mut offsets = [0_usize; 256];
    let mut sum = 0_usize;
    let indexed_histogram = histogram.iter().enumerate();
    for (i, &count) in indexed_histogram {
        offsets[i] = sum;
        sum += count as usize;
    }

    for &x in src {
        let raw = (x as u64) >> shift;
        let bucket = match sign_mode {
            SignBitMode::FlipMostSignificantBit => ((raw ^ 0x80) & 0xff) as usize,
            SignBitMode::Preserve => (raw & 0xff) as usize,
        };
        dst[offsets[bucket]] = x;
        offsets[bucket] += 1;
    }
}

/// Sorts a mutable slice of 64-bit signed integers (`i64`) in ascending order using
/// dual-sentinel L1 hash aggregation and an 8-pass LSD radix sort with degenerate-byte skipping.
pub fn lattice_sort_i64(slice: &mut [i64]) {
    let len = slice.len();
    if len <= 128 {
        if len > 1 {
            slice.sort_unstable();
        }
        return;
    }

    if try_sentinel_hash_aggregation_i64(slice) {
        return;
    }

    let mut histograms = [[0_u32; 256]; 8];
    for &x in slice.iter() {
        let u = x as u64;
        histograms[0][(u & 0xff) as usize] += 1;
        histograms[1][((u >> 8) & 0xff) as usize] += 1;
        histograms[2][((u >> 16) & 0xff) as usize] += 1;
        histograms[3][((u >> 24) & 0xff) as usize] += 1;
        histograms[4][((u >> 32) & 0xff) as usize] += 1;
        histograms[5][((u >> 40) & 0xff) as usize] += 1;
        histograms[6][((u >> 48) & 0xff) as usize] += 1;
        histograms[7][((u >> 56) ^ 0x80) as usize] += 1;
    }

    let n32 = len as u32;
    let mut scratch = vec![0_i64; len];
    let mut in_scratch = false;

    let indexed_histograms = histograms.iter().enumerate();
    for (byte_index, histogram) in indexed_histograms {
        if histogram.contains(&n32) {
            continue;
        }
        let shift = byte_index * 8;
        let sign_mode = if byte_index == 7 {
            SignBitMode::FlipMostSignificantBit
        } else {
            SignBitMode::Preserve
        };
        if in_scratch {
            scatter_pass_i64(&scratch, slice, histogram, shift, sign_mode);
            in_scratch = false;
        } else {
            scatter_pass_i64(slice, &mut scratch, histogram, shift, sign_mode);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice.copy_from_slice(&scratch);
    }
}
