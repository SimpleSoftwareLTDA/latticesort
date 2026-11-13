//! CLI benchmark runner for `latticesort` at $N = 1{,}000{,}000$.

use latticesort::{candidate, harness};

fn main() {
    let n = 1_000_000;
    let report = harness::evaluate_candidate(candidate::lattice_sort_i32, n);

    println!("=== AUTORESEARCH BENCHMARK SUITE (N = {n}) ===");
    println!(
        "{:<28} | {:>12} | {:>12} | {:>10}",
        "Dataset", "Lattice (ms)", "Std (ms)", "Speedup"
    );
    println!("{}", "-".repeat(70));
    for case in &report.cases {
        println!(
            "{:<28} | {:>12.3} | {:>12.3} | {:>9.2}x",
            case.label, case.candidate_ms, case.std_ms, case.speedup
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "{:<28} | {:>12.3} | {:>12.3} | {:>9.2}x",
        "GEOMETRIC MEAN (SCORE)",
        report.candidate_geomean_ms,
        report.std_geomean_ms,
        report.overall_speedup
    );
}

#[cfg(test)]
mod tests {
    use latticesort::{candidate, harness};

    #[test]
    fn verify_candidate_correctness() {
        harness::verify_edge_cases(candidate::lattice_sort_i32);
    }
}
