pub mod lexer;

pub use lexer::{Token, Lexer};

/// Простая функция, принимающая строку и возвращающая строку
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
