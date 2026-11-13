//! `latticesort`: adaptive structural and radix sorting library for primitive integers and
//! IEEE-754 floating-point slices.
//!
//! # Example
//!
//! ```rust
//! use latticesort::{LatticeSortExt, parallel};
//!
//! let mut numbers: Vec<i64> = vec![42, -10, 0, 999_999_999_999, 42];
//! numbers.lattice_sort();
//! assert!(numbers.is_sorted());
//!
//! let mut batch: Vec<i32> = (0..1000).rev().collect();
//! parallel::parallel_lattice_sort(&mut batch);
//! assert!(batch.is_sorted());
//! ```

#![warn(missing_docs)]

pub mod candidate;
pub mod harness;
pub mod kernel_i64;
pub mod parallel;
pub mod slice_ext;

pub use candidate::{HashAggregationOutcome, lattice_sort_i32};
pub use kernel_i64::lattice_sort_i64;
pub use parallel::parallel_lattice_sort;
pub use slice_ext::{LatticeSortExt, LatticeSortable};
