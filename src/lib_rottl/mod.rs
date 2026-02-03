pub mod lexer;
pub mod parser;

#[cfg(test)]
mod tests;

pub use lexer::{Lexer, Token};
pub use parser::{
    Argument, CallbackFn, CallbackMap, Context, EnumMap, EvalError, EvalErrorKind, EvalResult,
    Executable, ParseError, Parser, ParserConfig, PathAccessor, PathResolver, Value,
};

/// Simple function that takes a string and returns a string
pub fn get(input: &str) -> String {
    format!("Received: {}", input)
}
