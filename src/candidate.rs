// Experiment 0 (Baseline): Metatron 13-Ring Multi-Pivot Scratch Sort

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
    metatron_recurse::<12>(data, &mut scratch);
}

fn metatron_recurse<const RING_PIVOTS: usize>(slice: &mut [i32], scratch: &mut [i32]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }
    if len <= 32 {
        insertion_sort(slice);
        return;
    }

    let mut samples = Vec::with_capacity(RING_PIVOTS);
    let step = (len / (RING_PIVOTS + 1)).max(1);
    let mut idx = step;
    while idx < len && samples.len() < RING_PIVOTS {
        samples.push(slice[idx]);
        idx += step;
    }
    insertion_sort(&mut samples);

    let mut pivots = [0_i32; RING_PIVOTS];
    let mut k = 0_usize;
    for &val in &samples {
        if k == 0 || val != pivots[k - 1] {
            pivots[k] = val;
            k += 1;
            if k == RING_PIVOTS {
                break;
            }
        }
    }

    let total_buckets = 2 * k + 1;
    let mut counts = vec![0_usize; total_buckets];
    for &val in slice.iter() {
        let b = locate_bucket(&pivots[..k], val);
        counts[b] += 1;
    }

    let mut offsets = vec![0_usize; total_buckets];
    let mut acc = 0_usize;
    for i in 0..total_buckets {
        offsets[i] = acc;
        acc += counts[i];
    }

    let mut cursors = offsets.clone();
    let dst = &mut scratch[..len];
    for &val in slice.iter() {
        let b = locate_bucket(&pivots[..k], val);
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
            );
        }
    }
}

#[inline]
fn locate_bucket(pivots: &[i32], val: i32) -> usize {
    let mut lo = 0_usize;
    let mut hi = pivots.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if val < pivots[mid] {
            hi = mid;
        } else if val > pivots[mid] {
            lo = mid + 1;
        } else {
            return 2 * mid + 1;
        }
    }
    2 * lo
}
