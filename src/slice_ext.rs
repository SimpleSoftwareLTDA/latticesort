//! Polymorphic trait (`LatticeSortable`) and slice extension trait (`LatticeSortExt`)
//! supporting `i32`, `u32`, `i64`, `u64`, `f32`, and `f64` without intermediate key-buffer
//! allocations.

use crate::{candidate, kernel_i64, parallel};

/// Trait implemented by primitive numeric types that can be sorted via order-preserving
/// bijective bit projections in `latticesort`.
pub trait LatticeSortable: Copy {
    /// Sorts a mutable slice of `Self` in ascending total order on a single core.
    fn sort_slice(slice: &mut [Self]);

    /// Sorts a mutable slice of `Self` in ascending total order across available CPU cores.
    fn par_sort_slice(slice: &mut [Self]);
}

/// Extension trait adding `.lattice_sort()` and `.par_lattice_sort()` to slices `[T]`
/// where `T: LatticeSortable`.
///
/// Carries the `~Ext` suffix in accordance with the Google Rust Style Guide for extension
/// traits implemented on foreign slice types.
pub trait LatticeSortExt {
    /// Sorts the slice in ascending order on the current thread.
    fn lattice_sort(&mut self);

    /// Sorts the slice in ascending order in parallel across available CPU cores.
    fn par_lattice_sort(&mut self);
}

impl<T: LatticeSortable> LatticeSortExt for [T] {
    #[inline]
    fn lattice_sort(&mut self) {
        T::sort_slice(self);
    }

    #[inline]
    fn par_lattice_sort(&mut self) {
        T::par_sort_slice(self);
    }
}

impl LatticeSortable for i32 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        candidate::lattice_sort_generic_32(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_32(slice);
    }
}

impl LatticeSortable for u32 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        candidate::lattice_sort_generic_32(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_32(slice);
    }
}

impl LatticeSortable for f32 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        candidate::lattice_sort_generic_32(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_32(slice);
    }
}

impl LatticeSortable for i64 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        kernel_i64::lattice_sort_generic_64(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_64(slice);
    }
}

impl LatticeSortable for u64 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        kernel_i64::lattice_sort_generic_64(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_64(slice);
    }
}

impl LatticeSortable for f64 {
    #[inline]
    fn sort_slice(slice: &mut [Self]) {
        kernel_i64::lattice_sort_generic_64(slice);
    }

    #[inline]
    fn par_sort_slice(slice: &mut [Self]) {
        parallel::parallel_lattice_sort_64(slice);
    }
}

#[cfg(test)]
mod tests {
    use super::LatticeSortExt;

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

        let mut values_parallel_f32: Vec<f32> = (0..250_000).rev().map(|x| x as f32).collect();
        let mut expected_parallel_f32 = values_parallel_f32.clone();
        expected_parallel_f32.sort_unstable_by(f32::total_cmp);
        values_parallel_f32.par_lattice_sort();
        assert_eq!(
            values_parallel_f32, expected_parallel_f32,
            "par_lattice_sort on 250k reversed f32 elements must match f32::total_cmp"
        );
    }
}
