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
                let normalize_probabilities = |prob, level_idx| arity.pow(level_idx) as f32 * prob;
                let levels = predictions.len();
                let mut predictions = predictions
                    .iter()
                    .enumerate()
                    .flat_map(|(level_idx, level_predictions)| {
                        level_predictions
                            .iter()
                            .map(|(bucket_id, prob)| {
                                (
                                    level_idx,
                                    *bucket_id,
                                    normalize_probabilities(*prob, level_idx as u32),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                predictions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                predictions.into_iter().take(*nprobe).fold(
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
