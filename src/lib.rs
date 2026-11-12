pub mod candidate;
pub mod harness;

/// Generalized Sacred-Natural Sorting Trait for primitive integers and IEEE-754 floats.
pub trait SacredSort {
    fn sacred_sort(&mut self);
}

impl SacredSort for [i32] {
    fn sacred_sort(&mut self) {
        candidate::candidate_sort(self);
    }
}

impl SacredSort for [u32] {
    fn sacred_sort(&mut self) {
        if self.len() <= 1 {
            return;
        }
        // Bijective order-preserving map u32 -> i32 via sign-bit flip
        for x in self.iter_mut() {
            *x ^= 0x8000_0000;
        }
        let mut buf: Vec<i32> = self.iter().map(|&u| u as i32).collect();
        candidate::candidate_sort(&mut buf);
        for (dst, &src) in self.iter_mut().zip(buf.iter()) {
            *dst = (src as u32) ^ 0x8000_0000;
        }
    }
}

impl SacredSort for [f32] {
    fn sacred_sort(&mut self) {
        if self.len() <= 1 {
            return;
        }
        // IEEE-754 total_cmp bijection onto signed i32
        let mut mapped: Vec<i32> = self
            .iter()
            .map(|&f| {
                let bits = f.to_bits();
                let mask = (((bits as i32) >> 31) as u32) & 0x7FFF_FFFF;
                (bits ^ mask) as i32
            })
            .collect();
        candidate::candidate_sort(&mut mapped);
        for (dst, &m) in self.iter_mut().zip(mapped.iter()) {
            let bits = m as u32;
            let mask = ((m >> 31) as u32) & 0x7FFF_FFFF;
            *dst = f32::from_bits(bits ^ mask);
        }
    }
}

impl SacredSort for [i64] {
    fn sacred_sort(&mut self) {
        sort_i64_slice(self);
    }
}

impl SacredSort for [u64] {
    fn sacred_sort(&mut self) {
        if self.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i64> = self
            .iter()
            .map(|&u| (u ^ 0x8000_0000_0000_0000) as i64)
            .collect();
        sort_i64_slice(&mut mapped);
        for (dst, &m) in self.iter_mut().zip(mapped.iter()) {
            *dst = (m as u64) ^ 0x8000_0000_0000_0000;
        }
    }
}

impl SacredSort for [f64] {
    fn sacred_sort(&mut self) {
        if self.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i64> = self
            .iter()
            .map(|&f| {
                let bits = f.to_bits();
                let mask = (((bits as i64) >> 63) as u64) & 0x7FFF_FFFF_FFFF_FFFF;
                (bits ^ mask) as i64
            })
            .collect();
        sort_i64_slice(&mut mapped);
        for (dst, &m) in self.iter_mut().zip(mapped.iter()) {
            let bits = m as u64;
            let mask = ((m >> 63) as u64) & 0x7FFF_FFFF_FFFF_FFFF;
            *dst = f64::from_bits(bits ^ mask);
        }
    }
}

const CRYSTAL_MULTS_64: [u64; 4] = [
    0x9E37_79B9_7F4A_7C15,
    0xBF58_476D_1CE4_E5B9,
    0x94D0_49BB_1331_11EB,
    0xD6E8_FEB8_6659_FD93,
];

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
fn crystal_hash_64(val: i64, mult: u64) -> usize {
    let u = val as u64;
    (((u ^ (u >> 30)).wrapping_mul(mult)) >> 54) as usize
}

fn try_nucleation_i64(slice: &mut [i64]) -> bool {
    let len = slice.len();
    if len <= 2048 {
        return false;
    }

    let mut probe = [0_i64; 128];
    let anchors = [len / 10, (len * 3) / 10, len / 2, (len * 7) / 10, len - 16];
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
    insertion_sort_i64(&mut probe);

    let mut unique_probe = [0_i64; 128];
    unique_probe[0] = probe[0];
    let mut u_len = 1_usize;
    for i in 1..128 {
        if probe[i] != probe[i - 1] {
            if u_len >= 80 {
                return false;
            }
            unique_probe[u_len] = probe[i];
            u_len += 1;
        }
    }

    let mut chosen_mult = CRYSTAL_MULTS_64[0];
    for &m in &CRYSTAL_MULTS_64 {
        let mut occupied = [false; 1024];
        let mut ok = true;
        for &k in &unique_probe[..u_len] {
            let s = crystal_hash_64(k, m);
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

    let d0 = 0_i64;
    let s0 = crystal_hash_64(d0, chosen_mult);
    let mut d1 = 1_i64;
    while crystal_hash_64(d1, chosen_mult) == s0 {
        d1 += 1;
    }

    let mut keys = [d0; 1024];
    keys[s0] = d1;
    let mut counts = [0_u32; 1024];
    let mut used = [false; 1024];
    let mut unique_count = 0_usize;

    for &k in &unique_probe[..u_len] {
        let s = crystal_hash_64(k, chosen_mult);
        if !used[s] {
            used[s] = true;
            keys[s] = k;
            unique_count += 1;
        }
    }

    for &val in slice.iter() {
        let s = crystal_hash_64(val, chosen_mult);
        if keys[s] == val {
            counts[s] += 1;
        } else {
            let mut slot = s;
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

    let mut nuclei = [(0_i64, 0_usize); 384];
    let mut n_idx = 0_usize;
    for s in 0..1024 {
        if used[s] && counts[s] > 0 {
            nuclei[n_idx] = (keys[s], counts[s] as usize);
            n_idx += 1;
        }
    }

    for i in 1..n_idx {
        let key = nuclei[i];
        let mut j = i;
        while j > 0 && nuclei[j - 1].0 > key.0 {
            nuclei[j] = nuclei[j - 1];
            j -= 1;
        }
        nuclei[j] = key;
    }

    let mut offset = 0_usize;
    for &(val, count) in &nuclei[..n_idx] {
        slice[offset..offset + count].fill(val);
        offset += count;
    }
    true
}

#[inline]
fn scatter_pass_i64(src: &[i64], dst: &mut [i64], hist: &[u32; 256], shift: usize, flip_sign: bool) {
    let mut offsets = [0_usize; 256];
    let mut sum = 0_usize;
    for (i, &c) in hist.iter().enumerate() {
        offsets[i] = sum;
        sum += c as usize;
    }

    for &x in src {
        let raw = (x as u64) >> shift;
        let b = if flip_sign {
            ((raw ^ 0x80) & 0xFF) as usize
        } else {
            (raw & 0xFF) as usize
        };
        dst[offsets[b]] = x;
        offsets[b] += 1;
    }
}

/// 64-bit Octa-Shell Spectral Sort with Fused 8x256 Histogram and Degenerate Shell Skipping.
/// Real-world 64-bit integers (timestamps, IDs, bounded values) automatically skip constant upper shells!
pub fn sort_i64_slice(slice: &mut [i64]) {
    let len = slice.len();
    if len <= 128 {
        if len > 1 {
            slice.sort_unstable();
        }
        return;
    }

    if try_nucleation_i64(slice) {
        return;
    }

    let mut h = [[0_u32; 256]; 8];
    for &x in slice.iter() {
        let u = x as u64;
        h[0][(u & 0xFF) as usize] += 1;
        h[1][((u >> 8) & 0xFF) as usize] += 1;
        h[2][((u >> 16) & 0xFF) as usize] += 1;
        h[3][((u >> 24) & 0xFF) as usize] += 1;
        h[4][((u >> 32) & 0xFF) as usize] += 1;
        h[5][((u >> 40) & 0xFF) as usize] += 1;
        h[6][((u >> 48) & 0xFF) as usize] += 1;
        h[7][((u >> 56) ^ 0x80) as usize] += 1;
    }

    let n32 = len as u32;
    let mut scratch = vec![0_i64; len];
    let mut in_scratch = false;

    for (shell, hist) in h.iter().enumerate() {
        if hist.contains(&n32) {
            continue;
        }
        let shift = shell * 8;
        let flip = shell == 7;
        if in_scratch {
            scatter_pass_i64(&scratch, slice, hist, shift, flip);
            in_scratch = false;
        } else {
            scatter_pass_i64(slice, &mut scratch, hist, shift, flip);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice.copy_from_slice(&scratch);
    }
}

use std::num::NonZeroUsize;
use std::thread;

/// Multi-Core Parallel Sacred-Natural Sort (Front 4).
/// Uses `std::thread::scope` with disjoint mutable bucket slices (`split_at_mut`),
/// requiring zero `unsafe` blocks and zero external dependencies.
pub fn sacred_sort_par(slice: &mut [i32]) {
    let len = slice.len();
    if len < 200_000 {
        candidate::candidate_sort(slice);
        return;
    }

    let threads = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(4)
        .min(16);
    if threads <= 1 {
        candidate::candidate_sort(slice);
        return;
    }

    if candidate::try_soliton_wave_collapse(slice) {
        return;
    }
    let (nucleated, span_exceeds_horizon) = candidate::try_phyllotaxis_nucleation(slice);
    if nucleated {
        return;
    }
    if !span_exceeds_horizon && candidate::try_barycentric_condensation(slice) {
        return;
    }

    // Step 1: Parallel MSB Histogram across `threads` chunks
    let chunk_size = len.div_ceil(threads);
    let mut local_hists = vec![[0_u32; 256]; threads];
    thread::scope(|s| {
        for (chunk, hist_out) in slice.chunks(chunk_size).zip(local_hists.iter_mut()) {
            s.spawn(move || {
                let (chunks4, rem) = chunk.as_chunks::<4>();
                for c in chunks4 {
                    hist_out[(((c[0] as u32) >> 24) ^ 0x80) as usize] += 1;
                    hist_out[(((c[1] as u32) >> 24) ^ 0x80) as usize] += 1;
                    hist_out[(((c[2] as u32) >> 24) ^ 0x80) as usize] += 1;
                    hist_out[(((c[3] as u32) >> 24) ^ 0x80) as usize] += 1;
                }
                for &x in rem {
                    hist_out[(((x as u32) >> 24) ^ 0x80) as usize] += 1;
                }
            });
        }
    });

    let mut h3 = [0_u32; 256];
    for lh in &local_hists {
        for b in 0..256 {
            h3[b] += lh[b];
        }
    }

    let mut scratch = vec![0_i32; len];
    let n32 = len as u32;
    if h3.contains(&n32) {
        scratch.copy_from_slice(slice);
        candidate::l2_local_3shell_sort(slice, &mut scratch);
        return;
    }

    // Step 2: Top-level MSB scatter into `scratch`
    candidate::scatter_pass::<24, true>(slice, &mut scratch, &h3);

    // Step 3: Split `slice` and `scratch` into disjoint mutable bucket pairs via safe `split_at_mut`
    let mut buckets: Vec<(&mut [i32], &mut [i32])> = Vec::with_capacity(256);
    let mut rem_slice = &mut slice[..];
    let mut rem_scratch = &mut scratch[..];
    for &c in &h3 {
        let count = c as usize;
        if count == 0 {
            continue;
        }
        let (s_head, s_tail) = rem_slice.split_at_mut(count);
        let (sc_head, sc_tail) = rem_scratch.split_at_mut(count);
        rem_slice = s_tail;
        rem_scratch = sc_tail;
        buckets.push((s_head, sc_head));
    }

    // Step 4: Partition the bucket list into balanced thread groups and sort in parallel in L2 cache
    let target_per_thread = len.div_ceil(threads);
    thread::scope(|s| {
        let mut rem_buckets = &mut buckets[..];
        while !rem_buckets.is_empty() {
            let mut acc_elems = 0_usize;
            let mut split_idx = 0_usize;
            for (i, pair) in rem_buckets.iter().enumerate() {
                acc_elems += pair.0.len();
                split_idx = i + 1;
                if acc_elems >= target_per_thread {
                    break;
                }
            }
            let (batch, rest) = rem_buckets.split_at_mut(split_idx);
            rem_buckets = rest;
            s.spawn(move || {
                for (s_bucket, sc_bucket) in batch {
                    let count = s_bucket.len();
                    if count == 1 {
                        s_bucket[0] = sc_bucket[0];
                    } else {
                        candidate::l2_local_3shell_sort(s_bucket, sc_bucket);
                    }
                }
            });
        }
    });
}
