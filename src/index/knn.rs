use crate::metrics::Metric;

pub struct KNNIndex {
    dims: usize,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    metric: Metric,
}

impl KNNIndex {
    pub fn new(dims: usize, metric: Metric, ids: Vec<String>, vectors: Vec<Vec<f32>>) -> Self {
        assert_eq!(ids.len(), vectors.len(), "ids and vectors row count must match");
        for vec in &vectors {
            assert_eq!(vec.len(), dims, "each vector must have the same number of dimensions as specified");
        }
        Self {
            dims,
            ids,
            vectors,
            metric,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Brute-force exact kNN. Returns top-k candidates (higher score is better).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        assert_eq!(query.len(), self.dims, "query vector must match index dimensions");
        let mut scores: Vec<(String, f32)> = self.ids.iter()
            .zip(self.vectors.iter())
            .map(|(id, vec)| (id.clone(), self.metric.score(query, vec)))
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    } 
}

