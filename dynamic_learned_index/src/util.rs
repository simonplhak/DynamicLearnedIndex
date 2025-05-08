pub(crate) fn tensor2vec(tensor: &tch::Tensor) -> Vec<f64> {
    tensor.try_into().unwrap()
}

pub(crate) fn vec2tensor(vec: Vec<f64>) -> tch::Tensor {
    tch::Tensor::from_slice(&vec).to_kind(tch::kind::Kind::Float)
}
