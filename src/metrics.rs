pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

impl Metric {
    pub fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dimension mismatch");
        match self {
            Metric::Euclidean => -a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt(),
            Metric::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                dot / (norm_a * norm_b)
            }
            Metric::DotProduct => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        }
    }
}
