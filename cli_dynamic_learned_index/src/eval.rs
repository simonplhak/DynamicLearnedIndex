use dynamic_learned_index::types::{Array, Id};
use dynamic_learned_index::{Index, SearchStrategy};
use indicatif::ProgressBar;
use log::info;
use measure_time_macro::log_time;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct EvalMetrics {
    pub total: u64,       // total number of queries
    pub recall_top1: f32, // how many true top-5 queries were found in retrieved 5-nn results
    pub recall_top5: f32,
    pub recall_top10: f32,
    pub elapsed_time: std::time::Duration,
}

#[log_time]
pub fn eval_queries(
    index: &Index,
    gt: &[Vec<Id>],
    queries: &[Array],
    search_strategy: SearchStrategy,
    use_progress: bool,
) -> EvalMetrics {
    assert!(queries.len() == gt.len());
    let max_k = 10;
    assert!(queries.len() > max_k);
    assert!(index.size() >= max_k);
    let bar = match use_progress {
        true => Some(ProgressBar::new(gt.len() as u64).with_message("Inserting queries")),
        false => None,
    };
    let start = std::time::Instant::now();
    let results: Vec<_> = queries
        .iter()
        .map(|query| {
            if let Some(bar) = &bar {
                bar.inc(1);
            }
            index.search(query, (max_k, search_strategy))
        })
        .collect();
    let elapsed_time = start.elapsed();
    let (recall_top1, recall_top5, recall_top10) = results
        .iter()
        .zip(gt.iter())
        .map(|(res, gt)| {
            assert!(
                res.len() == max_k,
                "Expected {} results, got {}",
                max_k,
                res.len()
            );
            let recall_at_k = |k: usize| {
                let hits = res
                    .iter()
                    .take(k)
                    .filter(|idx| gt.iter().take(k).any(|gt_idx| *idx == gt_idx))
                    .count();
                hits as f32 / k as f32
            };

            let recall_top1 = recall_at_k(1);
            let recall_top5 = recall_at_k(5);
            let recall_top10 = recall_at_k(10);
            (recall_top1, recall_top5, recall_top10)
        })
        .fold((0.0, 0.0, 0.0), |(top1, top5, top10), (x, y, z)| {
            (top1 + x, top5 + y, top10 + z)
        });
    let total = queries.len() as f32;
    if let Some(bar) = &bar {
        bar.finish();
    }
    EvalMetrics {
        total: total as u64,
        recall_top1: recall_top1 / total,
        recall_top5: recall_top5 / total,
        recall_top10: recall_top10 / total,
        elapsed_time,
    }
}

#[derive(Copy, Clone)]
pub struct ValidationOptions {
    pub validate_after_n: usize, // validate after each n-th query
    pub include_each_n: usize,   // include each n-th query in validation
}

#[log_time]
pub fn insert_all_data(
    index: &mut Index,
    queries: Vec<Array>,
    limit: Option<usize>,
    validation_options: Option<ValidationOptions>,
    start_from_one: bool,
    search_strategy: SearchStrategy,
) {
    let limit = limit.unwrap_or(queries.len());

    let mut validation_ids = Vec::new();
    let mut validation_queries = Vec::new();
    let bar = ProgressBar::new(limit as u64).with_message("Inserting queries");
    let range = match start_from_one {
        true => 1..=limit,
        false => 0..=limit,
    };
    range.zip(queries.into_iter()).for_each(|(id, query)| {
        if let Some(validation_options) = validation_options {
            if id > 0 && id % validation_options.validate_after_n == 0 {
                let metrics = eval_queries(
                    index,
                    &validation_ids,
                    &validation_queries,
                    search_strategy,
                    false,
                );
                info!(total = metrics.total, recall_top1=metrics.recall_top1; "validation_metrics");
            }
            if id % validation_options.include_each_n == 0 {
                validation_ids.push(vec![id as Id]);
                validation_queries.push(query.clone());
            }
        }
        index.insert(query, id as Id);
        bar.inc(1);
    });
    bar.finish();
}
