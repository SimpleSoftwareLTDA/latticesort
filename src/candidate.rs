// Experiment 01: Kirkwood Coprime Golden Sampling + Cached Bucket Indices (Single-Lookup Metatron-13)

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

pub fn candidate_sort(data: &mut [i32]) {
    if data.len() <= 1 {
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

    // Golden-ratio quasi-random sampling (avoids rational resonance on periodic inputs)
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

    // Pass 1: Locate bucket once per element and cache in `tags`
    for (i, &val) in slice.iter().enumerate() {
        let b = locate_bucket_branchless(active_pivots, val);
        tags[i] = b as u8;
        counts[b] += 1;
    }

    // Fast path: if all elements matched a single equality bucket, slice is already uniform!
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
