// Experiment 10: Branchless CMOV Soliton Merge + 4x Unrolled Constant-Shift Tetra-Shell Spectral

const BARYCENTRIC_HORIZON_MAX: u64 = 1_048_576;

const CRYSTAL_MULTS: [u32; 4] = [
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

fn try_soliton_wave_collapse(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len < 32 {
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

    let mut tail = peak + 1;
    while tail + 1 < len && slice[tail] >= slice[tail + 1] {
        tail += 1;
    }
    if tail != len - 1 {
        return false;
    }

    if peak == 0 {
        slice.reverse();
        return true;
    }

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
    true
}

#[inline]
fn crystal_hash(val: i32, mult: u32) -> usize {
    let u = val as u32;
    (((u ^ (u >> 15)).wrapping_mul(mult)) >> 23) as usize
}

fn try_phyllotaxis_nucleation(slice: &mut [i32]) -> bool {
    let len = slice.len();
    if len < 128 {
        return false;
    }

    let mut probe = [0_i32; 128];
    let step = len / 128;
    for (i, slot) in probe.iter_mut().enumerate() {
        *slot = slice[i * step];
    }
    insertion_sort(&mut probe);

    let mut unique_probe = [0_i32; 64];
    unique_probe[0] = probe[0];
    let mut u_len = 1_usize;
    for i in 1..128 {
        if probe[i] != probe[i - 1] {
            if u_len >= 32 {
                return false;
            }
            unique_probe[u_len] = probe[i];
            u_len += 1;
        }
    }

    let mut chosen_mult = CRYSTAL_MULTS[0];
    for &m in &CRYSTAL_MULTS {
        let mut occupied = [false; 512];
        let mut ok = true;
        for &k in &unique_probe[..u_len] {
            let s = crystal_hash(k, m);
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
    let s0 = crystal_hash(d0, chosen_mult);
    let mut d1 = 1_i32;
    while crystal_hash(d1, chosen_mult) == s0 {
        d1 += 1;
    }

    let mut keys = [d0; 512];
    keys[s0] = d1;
    let mut counts = [0_u32; 512];
    let mut used = [false; 512];
    let mut unique_count = 0_usize;

    for &k in &unique_probe[..u_len] {
        let s = crystal_hash(k, chosen_mult);
        if !used[s] {
            used[s] = true;
            keys[s] = k;
            unique_count += 1;
        }
    }

    for &val in slice.iter() {
        let s = crystal_hash(val, chosen_mult);
        if keys[s] == val {
            counts[s] += 1;
        } else {
            let mut slot = s;
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
                slot = (slot + 1) & 511;
            }
        }
    }

    let mut nuclei = [(0_i32, 0_usize); 64];
    let mut idx = 0_usize;
    for s in 0..512 {
        if used[s] && counts[s] > 0 {
            nuclei[idx] = (keys[s], counts[s] as usize);
            idx += 1;
        }
    }

    for i in 1..idx {
        let key = nuclei[i];
        let mut j = i;
        while j > 0 && nuclei[j - 1].0 > key.0 {
            nuclei[j] = nuclei[j - 1];
            j -= 1;
        }
        nuclei[j] = key;
    }

    let mut offset = 0_usize;
    for &(val, count) in &nuclei[..idx] {
        slice[offset..offset + count].fill(val);
        offset += count;
    }
    true
}

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

#[inline]
fn scatter_pass<const SHIFT: usize, const FLIP_SIGN: bool>(
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

fn spectral_tetra_shell_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len <= 64 {
        insertion_sort(slice);
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
    let mut scratch = vec![0_i32; len];
    let mut in_scratch = false;

    if !h0.contains(&n32) {
        scatter_pass::<0, false>(slice, &mut scratch, &h0);
        in_scratch = true;
    }
    if !h1.contains(&n32) {
        if in_scratch {
            scatter_pass::<8, false>(&scratch, slice, &h1);
            in_scratch = false;
        } else {
            scatter_pass::<8, false>(slice, &mut scratch, &h1);
            in_scratch = true;
        }
    }
    if !h2.contains(&n32) {
        if in_scratch {
            scatter_pass::<16, false>(&scratch, slice, &h2);
            in_scratch = false;
        } else {
            scatter_pass::<16, false>(slice, &mut scratch, &h2);
            in_scratch = true;
        }
    }
    if !h3.contains(&n32) {
        if in_scratch {
            scatter_pass::<24, true>(&scratch, slice, &h3);
            in_scratch = false;
        } else {
            scatter_pass::<24, true>(slice, &mut scratch, &h3);
            in_scratch = true;
        }
    }

    if in_scratch {
        slice.copy_from_slice(&scratch);
    }
}

pub fn candidate_sort(data: &mut [i32]) {
    if data.len() <= 1 {
        return;
    }
    if try_soliton_wave_collapse(data) {
        return;
    }
    if try_phyllotaxis_nucleation(data) {
        return;
    }
    if try_barycentric_condensation(data) {
        return;
    }
    spectral_tetra_shell_sort(data);
}
