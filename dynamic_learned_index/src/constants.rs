const SIMD_REGISTER_SIZE: usize = 256;
pub const LANES: usize = SIMD_REGISTER_SIZE / 32;
pub const DEFAULT_BUCKET_SIZE: usize = 5_000;
pub const DEFAULT_BUFFER_SIZE: usize = 5_000;
pub const DEFAULT_INPUT_SHAPE: usize = 768;
pub const DEFAULT_ARITY: usize = 3;
pub const DEFAULT_SEARCH_N_CANDIDATES: usize = 30_000;
pub const DEFAULT_SEARCH_K: usize = 10;
