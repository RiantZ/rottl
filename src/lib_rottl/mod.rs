pub mod lexer;

pub use lexer::{Token, Lexer};

/// Simple function that takes a string and returns a string
pub fn get(input: &str) -> String {
    format!("Received: {}", input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get() {
        let result = get("hello");
        assert_eq!(result, "Received: hello");
    }
}
