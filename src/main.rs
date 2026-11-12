mod candidate;
mod harness;

fn main() {
    let n = 1_000_000;
    let report = harness::evaluate_candidate(candidate::candidate_sort, n);

    println!("=== AUTORESEARCH BENCHMARK SUITE (N = {n}) ===");
    println!(
        "{:<28} | {:>12} | {:>12} | {:>10}",
        "Dataset", "Cand (ms)", "Std (ms)", "Speedup"
    );
    println!("{}", "-".repeat(70));
    for c in &report.cases {
        println!(
            "{:<28} | {:>12.3} | {:>12.3} | {:>9.2}x",
            c.label, c.candidate_ms, c.std_ms, c.speedup
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
    use super::{candidate, harness};

    #[test]
    fn verify_candidate_correctness() {
        harness::verify_edge_cases(candidate::candidate_sort);
    }
}
