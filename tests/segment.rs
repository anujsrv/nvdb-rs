use nvdb_rs::metrics::Metric;
use nvdb_rs::storage::segment::{SegmentWriter, SegmentReader};
use tempfile::TempDir;

#[test]
fn test_segment() {
    let base_dir = TempDir::new().expect("unable to create temporary working directory");
    let mut writer = SegmentWriter::new(base_dir.path(), "test_segment", 3, Metric::Euclidean);
    writer.push(1, &[1.0, 2.0, 3.0]);
    writer.push(2, &[4.0, 5.0, 6.0]);
    writer.push(3, &[7.0, 8.0, 9.0]);
    let segment_path = writer.flush().expect("failed to flush segment");

    // Verify files exist
    assert!(segment_path.join("ids.bin").exists());
    assert!(segment_path.join("vectors.f32").exists());
    assert!(segment_path.join("meta.json").exists());
    
    let reader = SegmentReader::open(segment_path.as_path()).expect("failed to open segment reader");

    let index = reader.load_index().expect("failed to load index");
    assert_eq!(index.len(), 3);
    assert_eq!(index.dims(), 3);
    assert_eq!(index.search(&[1.0, 2.0, 3.0], 1)[0].0, 1);
    assert_eq!(index.search(&[4.0, 5.0, 6.0], 1)[0].0, 2);
    assert_eq!(index.search(&[7.0, 8.0, 9.0], 1)[0].0, 3);
}
