// LatticeSort 32-bit Core Kernel:
// 1. `try_bitonic_run_merge`: O(n) single-pass merge for ascending/descending/bitonic runs and long sorted prefixes
// 2. `try_sentinel_hash_aggregation`: 1024-slot L1 dual-sentinel direct-mapped hash aggregation (up to 384 distinct keys)
// 3. `try_dense_range_count`: offset counting sort when value span fits within L2/L3 cache horizon
// 4. `radix_lsd_fused_sort` & `radix_msb_l2_hybrid_sort`: 4x8-bit fused-histogram radix sort with degenerate-byte skipping

const DENSE_RANGE_MAX_SPAN: u64 = 1_048_576;
const SMALL_SORT_CUTOFF: usize = 96;
const STACK_SCRATCH_MAX: usize = 2048;
const MSB_L2_PARTITION_THRESHOLD: usize = 1_500_000;

const SENTINEL_HASH_MULTS_32: [u32; 4] = [
    0x9E37_79B9,
    0x85EB_CA6B,
    0xC2B2_AE35,
    0x27D4_EB2F,
];

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

fn in_place_small_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len <= 24 {
        insertion_sort(slice);
        return;
    }
    let mid = len / 2;
    let last = len - 1;
    if slice[mid] < slice[0] {
        slice.swap(0, mid);
    }
    if slice[last] < slice[0] {
        slice.swap(0, last);
    }
    if slice[last] < slice[mid] {
        slice.swap(mid, last);
    }
    let pivot = slice[mid];
    let mut i = 0_usize;
    let mut j = last;
    loop {
        while slice[i] < pivot {
            i += 1;
        }
        while slice[j] > pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        slice.swap(i, j);
        i += 1;
        j -= 1;
    }
    in_place_small_sort(&mut slice[..=j]);
    in_place_small_sort(&mut slice[j + 1..]);
}

fn merge_sorted_halves(slice: &mut [i32], split: usize) {
    if slice[split - 1] <= slice[split] {
        return;
    }
    let c = slice[0];
    if c == slice[split - 1] {
        let p = slice[split..].partition_point(|&x| x < c);
        slice.copy_within(split..split + p, 0);
        slice[p..split + p].fill(c);
        return;
    }

    let r0 = slice[split];
    let trim_left = slice[..split].partition_point(|&x| x <= r0);
    let active = &mut slice[trim_left..];
    let adj_split = split - trim_left;

    let len = active.len();
    let mut merged = vec![0_i32; len];
    let (left, right) = active.split_at(adj_split);
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut k = 0_usize;
    while i < left.len() && j < right.len() {
        let a = left[i];
        let b = right[j];
        if a <= b {
            merged[k] = a;
            i += 1;
        } else {
            merged[k] = b;
            j += 1;
        }
        k += 1;
    }
    if i < left.len() {
        let rem = left.len() - i;
        merged[k..k + rem].copy_from_slice(&left[i..]);
    } else if j < right.len() {
        let rem = right.len() - j;
        merged[k..k + rem].copy_from_slice(&right[j..]);
    }
    active.copy_from_slice(&merged);
}

pub fn try_bitonic_run_merge(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
        return false;
    }

    let step = len / 8;
    let mut dir_changes = 0_usize;
    let mut prev_dir = 0_i8;
    for i in 1..8 {
        let a = slice[(i - 1) * step];
        let b = slice[i * step];
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
    while peak + 1 < len && slice[peak] <= slice[peak + 1] {
        peak += 1;
    }
    if peak == len - 1 {
        return true;
    }

    if peak == 0 {
        let mut valley = 0_usize;
        while valley + 1 < len && slice[valley] >= slice[valley + 1] {
            valley += 1;
        }
        if valley == len - 1 {
            slice.reverse();
            return true;
        }
        let mut tail = valley + 1;
        while tail + 1 < len && slice[tail] <= slice[tail + 1] {
            tail += 1;
        }
        if tail == len - 1 {
            slice[..=valley].reverse();
            merge_sorted_halves(slice, valley + 1);
            return true;
        }
        return false;
    }

    let mut tail = peak + 1;
    while tail + 1 < len && slice[tail] >= slice[tail + 1] {
        tail += 1;
    }
    if tail == len - 1 {
        let mut merged = vec![0_i32; len];
        let (left, right) = slice.split_at(peak + 1);
        let mut i = 0_usize;
        let mut j = right.len();
        let mut k = 0_usize;
        while i < left.len() && j > 0 {
            let a = left[i];
            let b = right[j - 1];
            if a <= b {
                merged[k] = a;
                i += 1;
            } else {
                merged[k] = b;
                j -= 1;
            }
            k += 1;
        }
        if i < left.len() {
            let rem = left.len() - i;
            merged[k..k + rem].copy_from_slice(&left[i..]);
        } else {
            while j > 0 {
                merged[k] = right[j - 1];
                j -= 1;
                k += 1;
            }
        }
        slice.copy_from_slice(&merged);
        return true;
    }

    if peak >= (len * 3) / 4 {
        let c = slice[0];
        if slice[peak / 2] == c {
            while peak > 0 && slice[peak] != c {
                peak -= 1;
            }
        }
        lattice_sort_i32(&mut slice[peak + 1..]);
        merge_sorted_halves(slice, peak + 1);
        return true;
    }

    false
}

#[inline]
fn sentinel_hash_1024(val: i32, mult: u32) -> usize {
    let u = val as u32;
    (((u ^ (u >> 15)).wrapping_mul(mult)) >> 22) as usize
}

/// Self-verifying dual-sentinel hash aggregation on stack (`1024` slots, up to `384` unique keys).
/// Returns `(handled, probe_span_exceeds_horizon)`.
pub fn try_sentinel_hash_aggregation(slice: &mut [i32]) -> (bool, bool) {
    let len = slice.len();
    if len <= STACK_SCRATCH_MAX {
        return (false, true);
    }

    let mut probe = [0_i32; 128];
    let anchors = [
        len / 10,
        (len * 3) / 10,
        len / 2,
        (len * 7) / 10,
        len - 16,
    ];
    for (w, &base) in anchors.iter().enumerate() {
        let dst_start = w * 16;
        probe[dst_start..dst_start + 16].copy_from_slice(&slice[base..base + 16]);
    }

    let stride = ((len as u64 * 2_654_435_761) >> 32).max(1) as usize | 1;
    let mut idx = len / 7;
    for slot in probe.iter_mut().skip(80) {
        *slot = slice[idx % len];
        idx = idx.wrapping_add(stride);
    }
    insertion_sort(&mut probe);

    let probe_span = (i64::from(probe[127]) - i64::from(probe[0])) as u64;
    let horizon = (len as u64).min(DENSE_RANGE_MAX_SPAN);
    let span_exceeds_horizon = probe_span > horizon;

    let mut unique_probe = [0_i32; 128];
    unique_probe[0] = probe[0];
    let mut u_len = 1_usize;
    for i in 1..128 {
        if probe[i] != probe[i - 1] {
            if u_len >= 80 {
                return (false, span_exceeds_horizon);
            }
            unique_probe[u_len] = probe[i];
            u_len += 1;
        }
    }

    let mut chosen_mult = SENTINEL_HASH_MULTS_32[0];
    for &m in &SENTINEL_HASH_MULTS_32 {
        let mut occupied = [false; 1024];
        let mut ok = true;
        for &k in &unique_probe[..u_len] {
            let s = sentinel_hash_1024(k, m);
            if occupied[s] {
                ok = false;
                break;
            }
            occupied[s] = true;
        }
        if ok {
            chosen_mult = m;
            break;
        }
    }

    let d0 = 0_i32;
    let s0 = sentinel_hash_1024(d0, chosen_mult);
    let mut d1 = 1_i32;
    while sentinel_hash_1024(d1, chosen_mult) == s0 {
        d1 += 1;
    }

    let mut keys = [d0; 1024];
    keys[s0] = d1;
    let mut counts = [0_u32; 1024];
    let mut used = [false; 1024];
    let mut unique_count = 0_usize;

    for &k in &unique_probe[..u_len] {
        let s = sentinel_hash_1024(k, chosen_mult);
        if !used[s] {
            used[s] = true;
            keys[s] = k;
            unique_count += 1;
        }
    }

    for &val in slice.iter() {
        let s = sentinel_hash_1024(val, chosen_mult);
        if keys[s] == val {
            counts[s] += 1;
        } else {
            let mut slot = s;
            loop {
                if !used[slot] {
                    if unique_count >= 384 {
                        return (false, span_exceeds_horizon);
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

    let mut entries = [(0_i32, 0_usize); 384];
    let mut n_idx = 0_usize;
    for s in 0..1024 {
        if used[s] && counts[s] > 0 {
            entries[n_idx] = (keys[s], counts[s] as usize);
            n_idx += 1;
        }
    }

    for i in 1..n_idx {
        let key = entries[i];
        let mut j = i;
        while j > 0 && entries[j - 1].0 > key.0 {
            entries[j] = entries[j - 1];
            j -= 1;
        }
        entries[j] = key;
    }

    let mut offset = 0_usize;
    for &(val, count) in &entries[..n_idx] {
        slice[offset..offset + count].fill(val);
        offset += count;
    }
    (true, span_exceeds_horizon)
}

pub fn try_dense_range_count(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
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

    let horizon = (len as u64).min(DENSE_RANGE_MAX_SPAN);
    if span > horizon {
        return false;
    }

    let table_len = (span + 1) as usize;
    let mut density = vec![0_u32; table_len];
    let base = i64::from(min_val);
    for &x in slice.iter() {
        let offset_idx = (i64::from(x) - base) as usize;
        density[offset_idx] += 1;
    }

    let mut offset = 0_usize;
    for (idx, &count) in density.iter().enumerate() {
        if count > 0 {
            let val = (base + idx as i64) as i32;
            let c = count as usize;
            slice[offset..offset + c].fill(val);
            offset += c;
        }
    }
    true
}

#[inline]
pub fn scatter_pass<const SHIFT: usize, const FLIP_SIGN: bool>(
    src: &[i32],
    dst: &mut [i32],
    hist: &[u32; 256],
) {
    let mut offsets = [0_usize; 256];
    let mut sum = 0_usize;
    for (i, &c) in hist.iter().enumerate() {
        offsets[i] = sum;
        sum += c as usize;
    }

    for &x in src {
        let raw = (x as u32) >> SHIFT;
        let b = if FLIP_SIGN {
            ((raw ^ 0x80) & 0xFF) as usize
        } else {
            (raw & 0xFF) as usize
        };
        dst[offsets[b]] = x;
        offsets[b] += 1;
    }
}

pub fn l2_local_3pass_radix_sort(slice_bucket: &mut [i32], scratch_bucket: &mut [i32]) {
    let len = slice_bucket.len();
    if len <= SMALL_SORT_CUTOFF {
        if len > 1 {
            in_place_small_sort(slice_bucket);
        }
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];

    let (chunks, rem) = scratch_bucket.as_chunks::<4>();
    for c in chunks {
        let u0 = c[0] as u32;
        let u1 = c[1] as u32;
        let u2 = c[2] as u32;
        let u3 = c[3] as u32;

        h0[(u0 & 0xFF) as usize] += 1;
        h1[((u0 >> 8) & 0xFF) as usize] += 1;
        h2[((u0 >> 16) & 0xFF) as usize] += 1;

        h0[(u1 & 0xFF) as usize] += 1;
        h1[((u1 >> 8) & 0xFF) as usize] += 1;
        h2[((u1 >> 16) & 0xFF) as usize] += 1;

        h0[(u2 & 0xFF) as usize] += 1;
        h1[((u2 >> 8) & 0xFF) as usize] += 1;
        h2[((u2 >> 16) & 0xFF) as usize] += 1;

        h0[(u3 & 0xFF) as usize] += 1;
        h1[((u3 >> 8) & 0xFF) as usize] += 1;
        h2[((u3 >> 16) & 0xFF) as usize] += 1;
    }
    for &x in rem {
        let u = x as u32;
        h0[(u & 0xFF) as usize] += 1;
        h1[((u >> 8) & 0xFF) as usize] += 1;
        h2[((u >> 16) & 0xFF) as usize] += 1;
    }

    let n32 = len as u32;
    let mut in_scratch = true;

    if !h0.contains(&n32) {
        scatter_pass::<0, false>(scratch_bucket, slice_bucket, &h0);
        in_scratch = false;
    }
    if !h1.contains(&n32) {
        if in_scratch {
            scatter_pass::<8, false>(scratch_bucket, slice_bucket, &h1);
            in_scratch = false;
        } else {
            scatter_pass::<8, false>(slice_bucket, scratch_bucket, &h1);
            in_scratch = true;
        }
    }
    if !h2.contains(&n32) {
        if in_scratch {
            scatter_pass::<16, false>(scratch_bucket, slice_bucket, &h2);
            in_scratch = false;
        } else {
            scatter_pass::<16, false>(slice_bucket, scratch_bucket, &h2);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice_bucket.copy_from_slice(scratch_bucket);
    }
}

pub fn radix_msb_l2_hybrid_sort(slice: &mut [i32]) {
    let len = slice.len();
    let mut h3 = [0_u32; 256];
    let (chunks, rem) = slice.as_chunks::<4>();
    for c in chunks {
        h3[(((c[0] as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((c[1] as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((c[2] as u32) >> 24) ^ 0x80) as usize] += 1;
        h3[(((c[3] as u32) >> 24) ^ 0x80) as usize] += 1;
    }
    for &x in rem {
        h3[(((x as u32) >> 24) ^ 0x80) as usize] += 1;
    }

    let mut scratch = vec![0_i32; len];
    let n32 = len as u32;
    if h3.contains(&n32) {
        scratch.copy_from_slice(slice);
        l2_local_3pass_radix_sort(slice, &mut scratch);
        return;
    }

    let mut bucket_starts = [0_usize; 256];
    let mut sum = 0_usize;
    for (i, &c) in h3.iter().enumerate() {
        bucket_starts[i] = sum;
        sum += c as usize;
    }

    scatter_pass::<24, true>(slice, &mut scratch, &h3);

    for b in 0..256 {
        let count = h3[b] as usize;
        if count == 0 {
            continue;
        }
        let start = bucket_starts[b];
        let end = start + count;
        if count == 1 {
            slice[start] = scratch[start];
        } else if count <= SMALL_SORT_CUTOFF {
            slice[start..end].copy_from_slice(&scratch[start..end]);
            in_place_small_sort(&mut slice[start..end]);
        } else {
            l2_local_3pass_radix_sort(&mut slice[start..end], &mut scratch[start..end]);
        }
    }
}

pub fn radix_lsd_fused_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len <= SMALL_SORT_CUTOFF {
        in_place_small_sort(slice);
        return;
    }
    if len >= MSB_L2_PARTITION_THRESHOLD {
        radix_msb_l2_hybrid_sort(slice);
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];
    let mut h3 = [0_u32; 256];

    let (chunks, rem) = slice.as_chunks::<4>();
    for c in chunks {
        let u0 = c[0] as u32;
        let u1 = c[1] as u32;
        let u2 = c[2] as u32;
        let u3 = c[3] as u32;

        h0[(u0 & 0xFF) as usize] += 1;
        h1[((u0 >> 8) & 0xFF) as usize] += 1;
        h2[((u0 >> 16) & 0xFF) as usize] += 1;
        h3[((u0 >> 24) ^ 0x80) as usize] += 1;

        h0[(u1 & 0xFF) as usize] += 1;
        h1[((u1 >> 8) & 0xFF) as usize] += 1;
        h2[((u1 >> 16) & 0xFF) as usize] += 1;
        h3[((u1 >> 24) ^ 0x80) as usize] += 1;

        h0[(u2 & 0xFF) as usize] += 1;
        h1[((u2 >> 8) & 0xFF) as usize] += 1;
        h2[((u2 >> 16) & 0xFF) as usize] += 1;
        h3[((u2 >> 24) ^ 0x80) as usize] += 1;

        h0[(u3 & 0xFF) as usize] += 1;
        h1[((u3 >> 8) & 0xFF) as usize] += 1;
        h2[((u3 >> 16) & 0xFF) as usize] += 1;
        h3[((u3 >> 24) ^ 0x80) as usize] += 1;
    }
    for &x in rem {
        let u = x as u32;
        h0[(u & 0xFF) as usize] += 1;
        h1[((u >> 8) & 0xFF) as usize] += 1;
        h2[((u >> 16) & 0xFF) as usize] += 1;
        h3[((u >> 24) ^ 0x80) as usize] += 1;
    }

    let n32 = len as u32;
    let mut stack_buf = [0_i32; STACK_SCRATCH_MAX];
    let mut heap_buf: Vec<i32>;
    let scratch: &mut [i32] = if len <= STACK_SCRATCH_MAX {
        &mut stack_buf[..len]
    } else {
        heap_buf = vec![0_i32; len];
        &mut heap_buf
    };
    let mut in_scratch = false;

    if !h0.contains(&n32) {
        scatter_pass::<0, false>(slice, scratch, &h0);
        in_scratch = true;
    }
    if !h1.contains(&n32) {
        if in_scratch {
            scatter_pass::<8, false>(scratch, slice, &h1);
            in_scratch = false;
        } else {
            scatter_pass::<8, false>(slice, scratch, &h1);
            in_scratch = true;
        }
    }
    if !h2.contains(&n32) {
        if in_scratch {
            scatter_pass::<16, false>(scratch, slice, &h2);
            in_scratch = false;
        } else {
            scatter_pass::<16, false>(slice, scratch, &h2);
            in_scratch = true;
        }
    }
    if !h3.contains(&n32) {
        if in_scratch {
            scatter_pass::<24, true>(scratch, slice, &h3);
            in_scratch = false;
        } else {
            scatter_pass::<24, true>(slice, scratch, &h3);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice.copy_from_slice(scratch);
    }
}

pub fn lattice_sort_i32(data: &mut [i32]) {
    if data.len() <= SMALL_SORT_CUTOFF {
        if data.len() > 1 {
            in_place_small_sort(data);
        }
        return;
    }
    if try_bitonic_run_merge(data) {
        return;
    }
    let (aggregated, span_exceeds_horizon) = try_sentinel_hash_aggregation(data);
    if aggregated {
        return;
    }
    if !span_exceeds_horizon && try_dense_range_count(data) {
        return;
    }
    radix_lsd_fused_sort(data);
}

#[inline]
pub fn candidate_sort(data: &mut [i32]) {
    lattice_sort_i32(data);
}
