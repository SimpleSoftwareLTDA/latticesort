// Experiment 03: Barycentric Event-Horizon Condensation + Phyllotaxis Nucleation + Metatron-13

const GOLDEN_HASH_32: u32 = 0x9E37_79B9;
const BARYCENTRIC_HORIZON_MAX: u64 = 1_048_576;

fn insertion_sort(slice: &mut [i32]) {
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

fn try_phyllotaxis_nucleation(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len < 64 {
        return false;
    }

    let mut probe = [0_i32; 64];
    let step = len / 64;
    for (i, slot) in probe.iter_mut().enumerate() {
        *slot = slice[i * step];
    }
    insertion_sort(&mut probe);
    let mut distinct_probe = 1_usize;
    for i in 1..64 {
        if probe[i] != probe[i - 1] {
            distinct_probe += 1;
        }
    }
    if distinct_probe > 28 {
        return false;
    }

    let mut keys = [0_i32; 128];
    let mut counts = [0_u32; 128];
    let mut used = [false; 128];
    let mut unique_count = 0_usize;

    for &val in slice.iter() {
        let mut slot = ((val as u32).wrapping_mul(GOLDEN_HASH_32) >> 25) as usize;
        loop {
            if !used[slot] {
                if unique_count >= 64 {
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
            slot = (slot + 1) & 127;
        }
    }

    let mut nuclei = [(0_i32, 0_usize); 64];
    let mut idx = 0_usize;
    for s in 0..128 {
        if used[s] {
            nuclei[idx] = (keys[s], counts[s] as usize);
            idx += 1;
        }
    }

    for i in 1..unique_count {
        let key = nuclei[i];
        let mut j = i;
        while j > 0 && nuclei[j - 1].0 > key.0 {
            nuclei[j] = nuclei[j - 1];
            j -= 1;
        }
        nuclei[j] = key;
    }

    let mut offset = 0_usize;
    for &(val, count) in &nuclei[..unique_count] {
        slice[offset..offset + count].fill(val);
        offset += count;
    }
    true
}

/// Condenses integers into discrete energy orbits relative to the ground state `min_val`
/// whenever the energy span fits within the Barycentric Event Horizon.
fn try_barycentric_condensation(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len < 64 {
        return false;
    }

    let mut min_val = slice[0];
    let mut max_val = slice[0];
    for &x in &slice[1..] {
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

    let horizon = (len as u64).min(BARYCENTRIC_HORIZON_MAX);
    if span > horizon {
        return false;
    }

    let table_len = (span + 1) as usize;
    let mut density = vec![0_u32; table_len];
    let base = i64::from(min_val);
    for &x in slice.iter() {
        let orbit = (i64::from(x) - base) as usize;
        density[orbit] += 1;
    }

    let mut offset = 0_usize;
    for (orbit, &count) in density.iter().enumerate() {
        if count > 0 {
            let val = (base + orbit as i64) as i32;
            let c = count as usize;
            slice[offset..offset + c].fill(val);
            offset += c;
        }
    }
    true
}

pub fn candidate_sort(data: &mut [i32]) {
    if data.len() <= 1 {
        return;
    }
    if try_phyllotaxis_nucleation(data) {
        return;
    }
    if try_barycentric_condensation(data) {
        return;
    }
    let mut scratch = vec![0_i32; data.len()];
    let mut tags = vec![0_u8; data.len()];
    metatron_recurse::<12>(data, &mut scratch, &mut tags);
}

fn metatron_recurse<const RING_PIVOTS: usize>(
    slice: &mut [i32],
    scratch: &mut [i32],
    tags: &mut [u8],
) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    if len <= 32 {
        insertion_sort(slice);
        return;
    }

    let sample_cap = RING_PIVOTS * 2;
    let mut samples = [0_i32; 24];
    let actual_samples = sample_cap.min(24);
    let mut acc_idx = len / 3;
    let stride = ((len as u64 * 2_654_435_761) >> 32).max(1) as usize | 1;
    for slot in samples.iter_mut().take(actual_samples) {
        *slot = slice[acc_idx % len];
        acc_idx = acc_idx.wrapping_add(stride);
    }
    insertion_sort(&mut samples[..actual_samples]);

    let mut pivots = [0_i32; RING_PIVOTS];
    let mut k = 0_usize;
    for &val in &samples[..actual_samples] {
        if k == 0 || val != pivots[k - 1] {
            pivots[k] = val;
            k += 1;
            if k == RING_PIVOTS {
                break;
            }
        }
    }

    let total_buckets = 2 * k + 1;
    let mut counts = [0_usize; 27];
    let active_pivots = &pivots[..k];

    for (i, &val) in slice.iter().enumerate() {
        let b = locate_bucket_branchless(active_pivots, val);
        tags[i] = b as u8;
        counts[b] += 1;
    }

    for b in 0..k {
        if counts[2 * b + 1] == len {
            return;
        }
    }

    let mut offsets = [0_usize; 27];
    let mut acc = 0_usize;
    for i in 0..total_buckets {
        offsets[i] = acc;
        acc += counts[i];
    }

    let mut cursors = offsets;
    let dst = &mut scratch[..len];
    for (i, &val) in slice.iter().enumerate() {
        let b = tags[i] as usize;
        dst[cursors[b]] = val;
        cursors[b] += 1;
    }

    slice.copy_from_slice(dst);

    for b in 0..=k {
        let bucket_idx = 2 * b;
        let start_off = offsets[bucket_idx];
        let count = counts[bucket_idx];
        if count > 1 {
            metatron_recurse::<RING_PIVOTS>(
                &mut slice[start_off..start_off + count],
                &mut scratch[start_off..start_off + count],
                &mut tags[start_off..start_off + count],
            );
        }
    }
}

#[inline]
fn locate_bucket_branchless(pivots: &[i32], val: i32) -> usize {
    let mut idx = 0_usize;
    for &p in pivots {
        idx += usize::from(val > p) + usize::from(val >= p);
    }
    idx
}
