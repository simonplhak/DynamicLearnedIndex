use crate::constants::DEFAULT_SEARCH_N_CANDIDATES;

#[derive(Debug, Copy, Clone)]
pub enum SearchStrategy {
    Base(usize), // todo rename to KnnDriven
    ModelDriven(usize),
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::ModelDriven(DEFAULT_SEARCH_N_CANDIDATES)
    }
}

impl SearchStrategy {
    pub fn buckets2visit(
        &self,
        predictions: Vec<Vec<(usize, f32, usize)>>,
        occupied: usize,
    ) -> Vec<Vec<usize>> {
        assert!(!predictions.is_empty(), "Predictions cannot be empty");
        match self {
            SearchStrategy::Base(_nprobe) => todo!(),
            SearchStrategy::ModelDriven(ncandidates) => {
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
                            .map(|(bucket_id, prob, occupied)| {
                                (
                                    level_idx,
                                    *bucket_id,
                                    normalize_probability(*prob, level_idx as u32),
                                    *occupied,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                buckets2visit.sort_by(|a, b| b.2.total_cmp(&a.2));
                let mut res = vec![vec![]; levels];
                if occupied > *ncandidates {
                    return res;
                }
                let mut total_occupied = occupied;
                for (level_idx, bucket_id, _prob, occupied) in buckets2visit {
                    if occupied > 0 && total_occupied < *ncandidates {
                        res[level_idx].push(bucket_id);
                        total_occupied += occupied;
                    }
                    if total_occupied >= *ncandidates {
                        break;
                    }
                }
                res
            }
        }
    }
}
