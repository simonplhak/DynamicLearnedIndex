use dynamic_learned_index::Index;
use log::info;
use measure_time_macro::log_time;
use serde::Deserialize;
use tch::{IndexOp, Tensor};

#[derive(Deserialize)]
pub struct EvalMetrics {
    pub total: u64,       // total number of queries
    pub recall_top1: f32, // how many true top-5 queries were found in retrieved 5-nn results
    pub recall_top5: f32,
    pub recall_top10: f32,
}

pub fn eval_queries(index: &Index, gt: Tensor, queries: Tensor) -> EvalMetrics {
    assert!(queries.size()[0] == gt.size()[0]);
    let max_k = 10;
    let (recall_top1, recall_top5, recall_top10) = (0..queries.size()[0])
        .map(|i| {
            let tensor = queries.i((i, ..));
            let res = index.search(&tensor, max_k);
            assert!(res.len() == max_k);
            (i, res)
        })
        .map(|(i, res)| {
            let gt = gt.i((i, ..));
            let recall_at_k = |k: usize| {
                let gt = gt.narrow(0, 0, k as i64);
                let gt: Vec<i64> = gt.try_into().unwrap();
                let hits = res
                    .iter()
                    .take(k)
                    .filter(|idx| gt.contains(&(**idx as i64)))
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
    let total = queries.size()[0] as f32;
    EvalMetrics {
        total: total as u64,
        recall_top1: recall_top1 / total,
        recall_top5: recall_top5 / total,
        recall_top10: recall_top10 / total,
    }
}

#[log_time]
pub fn insert_all_data(index: &mut Index, data: Tensor) {
    (0..data.size()[0]).for_each(|i| {
        let tensor = data.i((i, ..));
        index.insert(tensor, i as u32);
    });
}
