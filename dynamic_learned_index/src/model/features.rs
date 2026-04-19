#[cfg(all(feature = "tch", feature = "candle", feature = "mix"))]
compile_error!("features \"tch\", \"candle\", \"mix\" are mutually exclusive — enable only one");

#[cfg(not(any(feature = "tch", feature = "candle", feature = "mix")))]
compile_error!("either \"tch\", \"candle\", \"mix\" feature must be enabled");
