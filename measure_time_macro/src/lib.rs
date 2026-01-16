use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn log_time(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let block = &input.block;
    let vis = &input.vis;
    let sig = &input.sig;

    let output = quote! {
        #vis #sig {
            let start = std::time::Instant::now();
            let result = (|| #block)();
            let duration = start.elapsed();
            debug!(time:?=duration, function=stringify!(#fn_name); "time");
            result
        }
    };

    output.into()
}
