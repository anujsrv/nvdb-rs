pub mod metrics;
pub mod index;
pub mod storage;

pub struct NvDB;

impl NvDB {
    pub fn new() -> Self {
        NvDB {}
    }
}
