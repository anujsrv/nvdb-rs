use crate::metrics::Metric;
use crate::storage::segment::SegmentWriter;
use kvs::{KvStore,KvsEngine};
use std::path::PathBuf;
use uuid::Uuid;

const FLUSH_THRESHOLD: usize = 1024 * 1024 * 1024; // in bytes

pub struct KNNIndex {
    dims: usize,
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    metric: Metric,
    kvstore: KvStore<u64, Vec<f32>>,
}

impl KNNIndex {
    pub fn new(dims: usize, metric: Metric, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> Self {
        assert_eq!(ids.len(), vectors.len(), "ids and vectors row count must match");
        for vec in &vectors {
            assert_eq!(vec.len(), dims, "each vector must have the same number of dimensions as specified");
        }
        let store_dir: PathBuf = "knn_kvstore".into();
        let kvstore = KvStore::open(&store_dir).unwrap();
        Self {
            dims,
            ids,
            vectors,
            metric,
            kvstore,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn add(&mut self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims, "vector must match index dimensions");
        self.kvstore.set(id.clone(), vector.clone()).unwrap();
        self.ids.push(id);
        self.vectors.push(vector);

        if self.len() * self.dims() * std::mem::size_of::<(u64, Vec<f32>)>() > FLUSH_THRESHOLD {
            self.write_to_segment();
        }
    }

    /// Brute-force exact kNN. Returns top-k candidates (higher score is better).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dims, "query vector must match index dimensions");
        let mut scores: Vec<(u64, f32)> = self.ids.iter()
            .zip(self.vectors.iter())
            .map(|(id, vec)| (id.clone(), self.metric.score(query, vec)))
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    } 

    pub fn write_to_segment(&mut self) {
        let segment_name = Uuid::new_v4().to_string();
        let dir: PathBuf = "segments".into();
        let mut segment = SegmentWriter::new(&dir, &segment_name, self.dims, self.metric); 
        for (id, vector) in self.ids.iter().zip(self.vectors.iter()) {
            segment.push(id.clone(), &vector);
        }
        segment.flush().unwrap();
    }
}
