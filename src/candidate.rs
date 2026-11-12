// Experiment 08: Tetra-Shell 8-8-8-8 L1-Resident Spectral Projection + Soliton + Phyllotaxis + Barycentric

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

    slice[peak + 1..].reverse();
    let mut merged = Vec::with_capacity(len);
    let (left, right) = slice.split_at(peak + 1);
    let mut i = 0_usize;
    let mut j = 0_usize;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            merged.push(left[i]);
            i += 1;
        } else {
            merged.push(right[j]);
            j += 1;
        }
    }
    if i < left.len() {
        merged.extend_from_slice(&left[i..]);
    }
    if j < right.len() {
        merged.extend_from_slice(&right[j..]);
    }
    slice.copy_from_slice(&merged);
    true
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

/// Tetra-Shell 8-8-8-8 L1-Resident Spectral Projection (256 write streams fit in L1 cache).
fn spectral_tetra_shell_sort(slice: &mut [i32]) {
    let len = slice.len();
    if len <= 64 {
        insertion_sort(slice);
        return;
    }

    let mut h = [[0_u32; 256]; 4];
    for &x in slice.iter() {
        let u = (x as u32) ^ 0x8000_0000;
        h[0][(u & 0xFF) as usize] += 1;
        h[1][((u >> 8) & 0xFF) as usize] += 1;
        h[2][((u >> 16) & 0xFF) as usize] += 1;
        h[3][(u >> 24) as usize] += 1;
    }

    let n32 = len as u32;
    let mut scratch = vec![0_i32; len];
    let mut in_scratch = false;

    for (shell, hist) in h.iter().enumerate() {
        if hist.contains(&n32) {
            continue;
        }
        let shift = shell * 8;
        let mut offsets = [0_usize; 256];
        let mut sum = 0_usize;
        for (i, &c) in hist.iter().enumerate() {
            offsets[i] = sum;
            sum += c as usize;
        }

        if in_scratch {
            for &x in &scratch {
                let u = ((x as u32) ^ 0x8000_0000) >> shift;
                let b = (u & 0xFF) as usize;
                slice[offsets[b]] = x;
                offsets[b] += 1;
            }
            in_scratch = false;
        } else {
            for &x in slice.iter() {
                let u = ((x as u32) ^ 0x8000_0000) >> shift;
                let b = (u & 0xFF) as usize;
                scratch[offsets[b]] = x;
                offsets[b] += 1;
            }
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
