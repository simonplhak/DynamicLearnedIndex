use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use dynamic_learned_index::{
    Array, CompactionStrategy, DeleteMethod, DistanceFn, Id, Index, IndexBuilder, ModelDevice,
    RebuildStrategy, SearchParams, SearchStrategy,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Path to YAML configuration file for the index
    pub index_config_path: Option<PathBuf>,
}

impl ServiceConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();

        let index_config_path = std::env::var("INDEX_CONFIG_PATH").ok().map(PathBuf::from);

        if index_config_path.is_some() {
            info!("Loaded INDEX_CONFIG_PATH from environment");
        }

        Self { index_config_path }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct InsertRequest {
    pub vector: Vec<f32>,
    pub id: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InsertResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

// ============================================================================
// Application State
// ============================================================================

pub struct AppState {
    index: Arc<Mutex<Index>>,
}

impl AppState {
    pub fn new(index: Index) -> Self {
        Self {
            index: Arc::new(Mutex::new(index)),
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /search - Search for vectors in the index
///
/// Request body:
/// ```json
/// {
///   "vector": [0.1, 0.2, 0.3, ...],
///   "k": 10
/// }
/// ```
pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ErrorResponse> {
    if req.vector.is_empty() {
        return Err(ErrorResponse {
            error: "Vector cannot be empty".to_string(),
        });
    }

    let query_slice: &[f32] = &req.vector;

    let index = state.index.lock().await;
    let results = index
        .search(
            query_slice,
            SearchParams {
                k: req.k,
                search_strategy: SearchStrategy::default(),
            },
        )
        .map_err(|e| {
            error!("Search error: {:?}", e);
            ErrorResponse {
                error: format!("Search failed: {}", e),
            }
        })?;

    Ok(Json(SearchResponse {
        results: results.into_iter().map(|id| id as u32).collect(),
    }))
}

/// POST /insert - Insert a vector in the index
///
/// Request body:
/// ```json
/// {
///   "vector": [0.1, 0.2, 0.3, ...],
///   "id": 1
/// }
/// ```
pub async fn insert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, ErrorResponse> {
    if req.vector.is_empty() {
        return Err(ErrorResponse {
            error: "Vector cannot be empty".to_string(),
        });
    }

    let array: Array = req.vector;

    let mut index = state.index.lock().await;
    index.insert(array, req.id as Id).map_err(|e| {
        error!("Insert error: {:?}", e);
        ErrorResponse {
            error: format!("Insert failed: {}", e),
        }
    })?;

    Ok(Json(InsertResponse {
        success: true,
        message: format!("Vector with id {} inserted successfully", req.id),
    }))
}

/// Health check endpoint
pub async fn health() -> &'static str {
    "OK"
}

// ============================================================================
// Main
// ============================================================================

/// Build index from config file or use defaults
fn build_index(config_path: Option<PathBuf>) -> Result<Index, Box<dyn std::error::Error>> {
    match config_path {
        Some(path) => {
            info!("Loading index configuration from: {:?}", path);
            if !path.exists() {
                return Err(format!("Config file not found: {:?}", path).into());
            }
            let index_builder = IndexBuilder::from_yaml(&path)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            info!("Index configuration loaded successfully");
            index_builder
                .build()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        None => {
            info!("No config file specified, using default configuration");
            // Create the index with default configuration
            // The vector dimension (input_shape) must match the dimension of inserted vectors
            IndexBuilder::default()
                .build()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Loading service configuration...");
    let config = ServiceConfig::from_env();

    info!("Building Dynamic Learned Index...");
    let index = build_index(config.index_config_path)?;

    info!("Index built successfully");

    let state = Arc::new(AppState::new(index));

    // Build router
    let app = Router::new()
        .route("/search", post(search))
        .route("/insert", post(insert))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;

    info!("Server listening on http://127.0.0.1:3000");

    axum::serve(listener, app).await?;

    Ok(())
}
