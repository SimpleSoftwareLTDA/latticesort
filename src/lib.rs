//! `latticesort`: adaptive structural and radix sorting library for primitive integers and
//! IEEE-754 floating-point slices.
//!
//! # Example
//!
//! ```rust
//! use latticesort::LatticeSortExt;
//!
//! let mut numbers: Vec<i64> = vec![42, -10, 0, 999_999_999_999, 42];
//! numbers.lattice_sort();
//! assert!(numbers.is_sorted());
//!
//! let mut batch: Vec<f32> = (0..1000).rev().map(|x| x as f32).collect();
//! batch.par_lattice_sort();
//! assert!(batch.is_sorted());
//! ```

#![warn(missing_docs)]

pub mod candidate;
#[doc(hidden)]
pub mod harness;
pub mod kernel_i64;
pub mod parallel;
pub mod slice_ext;

pub use candidate::{HashAggregationOutcome, LatticeKey32, lattice_sort_i32};
pub use kernel_i64::{LatticeKey64, lattice_sort_i64};
pub use parallel::{parallel_lattice_sort, parallel_lattice_sort_32, parallel_lattice_sort_64};
pub use slice_ext::{LatticeSortExt, LatticeSortable};
