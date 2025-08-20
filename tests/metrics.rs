use nvdb_rs::metrics::Metric;

#[test]
fn test_euclidean_score() {
    let metric = Metric::Euclidean;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let score = -metric.score(&a, &b); // since score is negative in the implementation
    assert!((score - 5.196152).abs() < 1e-6, "Euclidean distance calculation failed");
} 

#[test]
fn test_cosine_score() {
    let metric = Metric::Cosine;
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let score = metric.score(&a, &b);
    assert!((score - 0.0).abs() < 1e-6, "Cosine similarity calculation failed");
}

#[test]
fn test_dot_product_score() {
    let metric = Metric::DotProduct;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let score = -metric.score(&a, &b); // since score is negative in the implementation
    assert!((score - 32.0).abs() < 1e-6, "Dot product calculation failed");
}

