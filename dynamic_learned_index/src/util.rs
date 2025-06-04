use crate::types::{Array, ArraySlice};

pub(crate) fn tensor2vec(tensor: &tch::Tensor) -> Array {
    tensor.try_into().unwrap()
}

pub(crate) fn vec2tensor(vec: &ArraySlice) -> tch::Tensor {
    tch::Tensor::from_slice(vec).to_kind(tch::kind::Kind::Float)
}
