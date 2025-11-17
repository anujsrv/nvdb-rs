use crate::metrics::Metric;
use crate::index::knn::KNNIndex;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::io::{Read, Write, Error};
use serde_json;

pub struct SegmentWriter {
    dir: PathBuf,
    seg_name: String,
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

pub struct SegmentReader{
    dir: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Metadata {
    count: u32,
    dim: usize,
    metric: String,
}

impl SegmentWriter {
    pub fn new(base_dir: &Path, seg_name: &str, dim: usize, metric: Metric) -> Self {
        let tmp_dir = base_dir.join(format!("{}.tmp", seg_name));
        let _ = fs::create_dir_all(tmp_dir.as_path());
        Self {
            dir: tmp_dir,
            seg_name: seg_name.to_string(),
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub fn push(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.ids.push(id);
        self.vectors.extend(vector.iter());
    }
    
    pub fn flush(self) -> Result<PathBuf, Error> {
        let ids_path = self.dir.join("ids.bin");
        let vectors_path = self.dir.join("vectors.f32");
        let meta_path = self.dir.join("meta.json");

        let mut ids_file: File = File::create(ids_path.as_path())?;
        ids_file.write_all(self.ids
                  .iter()
                  .flat_map(|id| id.to_le_bytes())
                  .collect::<Vec<u8>>().as_slice())?;
        ids_file.sync_all()?;

        let mut vectors_file: File = File::create(vectors_path.as_path())?;
        vectors_file.write_all(self.vectors
                  .iter()
                  .flat_map(|v| v.to_le_bytes())
                  .collect::<Vec<u8>>().as_slice())?;
        vectors_file.sync_all()?;

        let mut meta_file: File = File::create(meta_path.as_path())?;
        let metadata = Metadata {
            count: self.ids.len() as u32,
            dim: self.dim,
            metric: format!("{:?}", self.metric),
        };
        meta_file.write_all(serde_json::to_string(&metadata)?.as_bytes())?;
        meta_file.sync_all()?;

        let final_dir = self.dir.parent().unwrap().join(&self.seg_name);
        fs::rename(self.dir.as_path(), final_dir.as_path())?;

        // call fsync on parent directory to ensure rename is persisted
        let parent_dir_file = File::open(final_dir.parent().unwrap())?;
        parent_dir_file.sync_all()?;

        Ok(final_dir)
    }
}

impl SegmentReader {
    pub fn open(dir: &Path) -> Result<Self, Error> {
        File::open(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    pub fn load_index(&self) -> Result<KNNIndex, Error> {
        let metadata: Metadata = self.meta()?;

        let ids_path = self.dir.join("ids.bin");
        let mut ids_file = File::open(ids_path.as_path())?;
        let mut ids_buf = Vec::new();
        ids_file.read_to_end(&mut ids_buf)?;
        let ids: Vec<u64> = ids_buf
            .chunks_exact(8)
            .map(|chunk| {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                u64::from_le_bytes(arr)
            })
            .collect();

        let vectors_path = self.dir.join("vectors.f32");
        let mut vectors_file = File::open(vectors_path.as_path())?;
        let mut vectors_buf = Vec::new();
        vectors_file.read_to_end(&mut vectors_buf)?;
        let vectors: Vec<f32> = vectors_buf
            .chunks_exact(4)
            .map(|chunk| {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(chunk);
                f32::from_le_bytes(arr)
            })
            .collect();

        if metadata.count as usize != ids.len() {
            return Err(Error::new(std::io::ErrorKind::InvalidData, "ID count mismatch"));
        }
        if metadata.count as usize * metadata.dim != vectors.len() {
            return Err(Error::new(std::io::ErrorKind::InvalidData, "Vector count mismatch"));
        }

        let metric = match metadata.metric.as_str() {
            "Euclidean" => Metric::Euclidean,
            "Cosine" => Metric::Cosine,
            "DotProduct" => Metric::DotProduct,
            _ => return Err(Error::new(std::io::ErrorKind::InvalidData, "Unknown metric")),
        };

        Ok(KNNIndex::new(
            metadata.dim,
            metric,
            ids,
            vectors.chunks_exact(metadata.dim).map(|chunk| chunk.to_vec()).collect(),
        ))
    }

     pub fn meta(&self) -> Result<Metadata, Error> {
        let meta_path = self.dir.join("meta.json");
        let meta_file = File::open(meta_path.as_path())?;
        let metadata: Metadata = serde_json::from_reader(meta_file)?;

        Ok(metadata)
     }
}
