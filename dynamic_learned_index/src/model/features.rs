#[cfg(all(feature = "tch", feature = "candle"))]
compile_error!("features \"tch\" and \"candle\" are mutually exclusive — enable only one");

#[cfg(not(any(feature = "tch", feature = "candle")))]
compile_error!("either \"tch\" or \"candle\" feature must be enabled");
