//! Polymorphic trait (`LatticeSortable`) and slice extension trait (`LatticeSortExt`)
//! supporting `i32`, `u32`, `i64`, `u64`, `f32`, and `f64`.

use crate::{candidate, kernel_i64};

/// Trait implemented by primitive numeric types that can be sorted via order-preserving
/// bijective bit projections in `latticesort`.
pub trait LatticeSortable: Copy {
    /// Sorts a mutable slice of `Self` in ascending total order.
    fn sort_slice(slice: &mut [Self]);
}

/// Extension trait adding `.lattice_sort()` to slices `[T]` where `T: LatticeSortable`.
///
/// Carries the `~Ext` suffix in accordance with the Google Rust Style Guide for extension
/// traits implemented on foreign slice types.
pub trait LatticeSortExt {
    /// Sorts the slice in ascending order.
    fn lattice_sort(&mut self);
}

impl<T: LatticeSortable> LatticeSortExt for [T] {
    #[inline]
    fn lattice_sort(&mut self) {
        T::sort_slice(self);
    }
}

impl LatticeSortable for i32 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        candidate::lattice_sort_i32(slice);
    }
}

impl LatticeSortable for u32 {
    fn sort_slice(slice: &mut [Self]) {
        if slice.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i32> = slice
            .iter()
            .map(|&val| (val ^ 0x8000_0000) as i32)
            .collect();
        candidate::lattice_sort_i32(&mut mapped);
        let pairs = slice.iter_mut().zip(mapped.iter());
        for (dst, &src) in pairs {
            *dst = (src as u32) ^ 0x8000_0000;
        }
    }
}

impl LatticeSortable for f32 {
    fn sort_slice(slice: &mut [Self]) {
        if slice.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i32> = slice
            .iter()
            .map(|&val| {
                let bits = val.to_bits();
                let mask = (((bits as i32) >> 31) as u32) & 0x7fff_ffff;
                (bits ^ mask) as i32
            })
            .collect();
        candidate::lattice_sort_i32(&mut mapped);
        let pairs = slice.iter_mut().zip(mapped.iter());
        for (dst, &encoded) in pairs {
            let bits = encoded as u32;
            let mask = ((encoded >> 31) as u32) & 0x7fff_ffff;
            *dst = f32::from_bits(bits ^ mask);
        }
    }
}

impl LatticeSortable for i64 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        kernel_i64::lattice_sort_i64(slice);
    }
}

impl LatticeSortable for u64 {
    fn sort_slice(slice: &mut [Self]) {
        if slice.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i64> = slice
            .iter()
            .map(|&val| (val ^ 0x8000_0000_0000_0000) as i64)
            .collect();
        kernel_i64::lattice_sort_i64(&mut mapped);
        let pairs = slice.iter_mut().zip(mapped.iter());
        for (dst, &encoded) in pairs {
            *dst = (encoded as u64) ^ 0x8000_0000_0000_0000;
        }
    }
}

impl LatticeSortable for f64 {
    fn sort_slice(slice: &mut [Self]) {
        if slice.len() <= 1 {
            return;
        }
        let mut mapped: Vec<i64> = slice
            .iter()
            .map(|&val| {
                let bits = val.to_bits();
                let mask = (((bits as i64) >> 63) as u64) & 0x7fff_ffff_ffff_ffff;
                (bits ^ mask) as i64
            })
            .collect();
        kernel_i64::lattice_sort_i64(&mut mapped);
        let pairs = slice.iter_mut().zip(mapped.iter());
        for (dst, &encoded) in pairs {
            let bits = encoded as u64;
            let mask = ((encoded >> 63) as u64) & 0x7fff_ffff_ffff_ffff;
            *dst = f64::from_bits(bits ^ mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LatticeSortExt;
    use crate::parallel;

    #[test]
    fn test_generic_lattice_sort_all_types() {
        let mut values_u32 = vec![400_000_000_u32, 0, u32::MAX, 42, 7, 7];
        let mut expected_u32 = values_u32.clone();
        expected_u32.sort_unstable();
        values_u32.lattice_sort();
        assert_eq!(
            values_u32, expected_u32,
            "u32 slice after lattice_sort must match std::slice::sort_unstable"
        );

        let mut values_i64 = vec![i64::MAX, i64::MIN, 0, -42, 42, -1];
        let mut expected_i64 = values_i64.clone();
        expected_i64.sort_unstable();
        values_i64.lattice_sort();
        assert_eq!(
            values_i64, expected_i64,
            "i64 slice after lattice_sort must match std::slice::sort_unstable"
        );

        let mut values_u64 = vec![u64::MAX, 0, 123_456_789_012_345, 42, 42];
        let mut expected_u64 = values_u64.clone();
        expected_u64.sort_unstable();
        values_u64.lattice_sort();
        assert_eq!(
            values_u64, expected_u64,
            "u64 slice after lattice_sort must match std::slice::sort_unstable"
        );

        let mut values_f32 = vec![
            3.5_f32,
            -0.0,
            0.0,
            -100.25,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut expected_f32 = values_f32.clone();
        expected_f32.sort_unstable_by(f32::total_cmp);
        values_f32.lattice_sort();
        let pairs_f32 = values_f32.iter().zip(expected_f32.iter());
        for (actual, expected) in pairs_f32 {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f32 IEEE-754 total ordering (including -0.0 and +0.0) must match f32::total_cmp"
            );
        }

        let mut values_f64 = vec![
            1.25_f64,
            -0.0,
            0.0,
            -9999.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut expected_f64 = values_f64.clone();
        expected_f64.sort_unstable_by(f64::total_cmp);
        values_f64.lattice_sort();
        let pairs_f64 = values_f64.iter().zip(expected_f64.iter());
        for (actual, expected) in pairs_f64 {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f64 IEEE-754 total ordering (including -0.0 and +0.0) must match f64::total_cmp"
            );
        }

        let mut values_parallel: Vec<i32> = (0..250_000).rev().collect();
        let mut expected_parallel = values_parallel.clone();
        expected_parallel.sort_unstable();
        parallel::parallel_lattice_sort(&mut values_parallel);
        assert_eq!(
            values_parallel, expected_parallel,
            "parallel_lattice_sort on 250k reversed elements must match std::slice::sort_unstable"
        );
    }
}
