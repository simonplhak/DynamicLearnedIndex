pub enum SearchStrategy {
    Base(BaseSearchConfig),
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::Base(BaseSearchConfig::default())
    }
}

impl SearchStrategy {
    pub fn nprobe(&self) -> usize {
        match self {
            SearchStrategy::Base(config) => config.nprobe,
        }
    }
}

pub struct BaseSearchConfig {
    nprobe: usize,
}

impl Default for BaseSearchConfig {
    fn default() -> Self {
        BaseSearchConfig { nprobe: 1 }
    }
}
