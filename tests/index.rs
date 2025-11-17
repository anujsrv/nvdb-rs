use nvdb_rs::index::knn::KNNIndex;
use nvdb_rs::metrics::Metric;

#[test]
fn test_new() {
    let ids = vec![1, 2];
    let vectors = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let index = KNNIndex::new(2, Metric::Cosine, ids.clone(), vectors.clone());
    assert_eq!(index.len(), 2);
    assert_eq!(index.dims(), 2);
}

#[test]
fn test_search() {
    let ids = vec![1, 2, 3];
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]];

    // cosine similarity
    let index = KNNIndex::new(2, Metric::Cosine, ids.clone(), vectors.clone());
    let query = vec![1.0, 0.0];
    let results = index.search(&query, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 3);

    // euclidean distance
    let index = KNNIndex::new(2, Metric::Euclidean, ids.clone(), vectors.clone());
    let query = vec![1.0, 0.0];
    let results = index.search(&query, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 3);
}
