use async_stream::stream;
use axum::{
    response::sse::{Event, Sse},
    routing::get,
    Router,
};
use futures::Stream;
use tokio::time::sleep;

use std::{
    convert::Infallible,
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::PathBuf,
    time::Duration,
};
use tower_http::{services::ServeDir, trace::TraceLayer};

#[tokio::main]
async fn main() {
    let app = app();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn app() -> Router {
    let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let static_files_service = ServeDir::new(assets_dir).append_index_html_on_directories(true);
    Router::new()
        .fallback_service(static_files_service)
        .route("/sse", get(sse_handler))
        .layer(TraceLayer::new_for_http())
}

async fn sse_handler() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    println!("connected");
    let stream = stream! {
        let mut file = File::open(
                "/home/simon/Documents/dip/DynamicLearnedIndex/experiments_data/test/logs.jsonl",
            )
            .unwrap();
        let mut pos = file.stream_position().unwrap();
        loop {
            sleep(Duration::from_secs(1)).await;

            file.seek(SeekFrom::Start(pos)).unwrap();
            let reader = BufReader::new(&file);

            for line in reader.lines().map_while(Result::ok) {
                yield Ok(Event::default().json_data(line).unwrap());
            }

            pos = file.stream_position().unwrap();
        }
    };
    Sse::new(stream)
}
