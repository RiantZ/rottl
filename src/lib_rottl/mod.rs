pub mod lexer;
pub mod parser;

#[cfg(test)]
mod tests;

pub use lexer::{Lexer, Token};
pub use parser::{
    Argument, BoxError, CallbackFn, CallbackMap, EnumMap, EvalContext, Parser, PathAccessor,
    PathResolver, Result, Value,
};

/// Simple function that takes a string and returns a string
pub fn get(input: &str) -> String {
    format!("Received: {}", input)
}
