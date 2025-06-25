#[derive(Debug, Copy, Clone)]
pub enum SearchStrategy {
    Base(usize), // todo rename to KnnDriven
    ModelDriven(usize),
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::ModelDriven(1)
    }
}

impl SearchStrategy {
    pub fn buckets2visit(&self, predictions: Vec<Vec<(usize, f32)>>) -> Vec<Vec<usize>> {
        assert!(!predictions.is_empty(), "Predictions cannot be empty");
        match self {
            SearchStrategy::Base(nprobe) => predictions
                .iter()
                .map(|level_pred| {
                    level_pred
                        .iter()
                        .take(*nprobe)
                        .map(|(bucket_id, _)| *bucket_id)
                        .collect()
                })
                .collect(),
            SearchStrategy::ModelDriven(nprobe) => {
                let arity = predictions[0].len();
                let normalize_probability =
                    |prob: f32, level_idx| (arity.pow(level_idx) as f32) * prob.max(0.0);
                let levels = predictions.len();
                let mut buckets2visit = predictions
                    .iter()
                    .enumerate()
                    .flat_map(|(level_idx, level_predictions)| {
                        level_predictions
                            .iter()
                            .map(|(bucket_id, prob)| {
                                (
                                    level_idx,
                                    *bucket_id,
                                    normalize_probability(*prob, level_idx as u32),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                buckets2visit.sort_by(|a, b| b.2.total_cmp(&a.2));
                buckets2visit.into_iter().take(*nprobe).fold(
                    vec![vec![]; levels],
                    |mut acc, (level_idx, bucket_id, _)| {
                        acc[level_idx].push(bucket_id);
                        acc
                    },
                )
            }
        }
    }
}
