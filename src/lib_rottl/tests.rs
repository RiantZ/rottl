//! Tests for the OTTL lexer and library

use crate::lexer::{Lexer, Token};
use crate::parser::Parser;
use crate::{CallbackMap, EnumMap, EvalContext, OttlParser, PathAccessor, PathResolver, Value};
use std::sync::Arc;

// ============================================================================
// Lexer tests
// ============================================================================

#[test]
fn test_keywords() {
    let tokens = Lexer::collect_tokens("where or and not true false nil");
    assert_eq!(
        tokens,
        vec![
            Token::Where,
            Token::Or,
            Token::And,
            Token::Not,
            Token::True,
            Token::False,
            Token::Nil,
        ]
    );
}

#[test]
fn test_comparison_operators() {
    let tokens = Lexer::collect_tokens("== != < > <= >=");
    assert_eq!(
        tokens,
        vec![
            Token::Eq,
            Token::NotEq,
            Token::Less,
            Token::Greater,
            Token::LessEq,
            Token::GreaterEq,
        ]
    );
}

#[test]
fn test_arithmetic_operators() {
    let tokens = Lexer::collect_tokens("+ - * /");
    assert_eq!(
        tokens,
        vec![Token::Plus, Token::Minus, Token::Star, Token::Slash,]
    );
}

#[test]
fn test_delimiters() {
    let tokens = Lexer::collect_tokens("( ) [ ] { } , . : =");
    assert_eq!(
        tokens,
        vec![
            Token::LParen,
            Token::RParen,
            Token::LBracket,
            Token::RBracket,
            Token::LBrace,
            Token::RBrace,
            Token::Comma,
            Token::Dot,
            Token::Colon,
            Token::Assign,
        ]
    );
}

#[test]
fn test_string_literal() {
    let tokens = Lexer::collect_tokens(r#""hello world""#);
    assert_eq!(tokens, vec![Token::StringLiteral(r#""hello world""#)]);
}

#[test]
fn test_string_with_escape() {
    let tokens = Lexer::collect_tokens(r#""hello \"world\"""#);
    assert_eq!(tokens, vec![Token::StringLiteral(r#""hello \"world\"""#)]);
}

#[test]
fn test_int_literal() {
    // Note: Signs are now separate tokens, handled by the parser
    let tokens = Lexer::collect_tokens("42 0");
    assert_eq!(
        tokens,
        vec![Token::IntLiteral("42"), Token::IntLiteral("0"),]
    );
}

#[test]
fn test_signed_int_literal() {
    // Signs are separate tokens
    let tokens = Lexer::collect_tokens("-10 +5");
    assert_eq!(
        tokens,
        vec![
            Token::Minus,
            Token::IntLiteral("10"),
            Token::Plus,
            Token::IntLiteral("5"),
        ]
    );
}

#[test]
fn test_float_literal() {
    // Note: Signs are now separate tokens, handled by the parser
    let tokens = Lexer::collect_tokens("6.18 .5");
    assert_eq!(
        tokens,
        vec![Token::FloatLiteral("6.18"), Token::FloatLiteral(".5"),]
    );
}

#[test]
fn test_signed_float_literal() {
    // Signs are separate tokens
    let tokens = Lexer::collect_tokens("-2.0 +0.1");
    assert_eq!(
        tokens,
        vec![
            Token::Minus,
            Token::FloatLiteral("2.0"),
            Token::Plus,
            Token::FloatLiteral("0.1"),
        ]
    );
}

#[test]
fn test_bytes_literal() {
    let tokens = Lexer::collect_tokens("0xDEADBEEF 0x00 0xabc123");
    assert_eq!(
        tokens,
        vec![
            Token::BytesLiteral("0xDEADBEEF"),
            Token::BytesLiteral("0x00"),
            Token::BytesLiteral("0xabc123"),
        ]
    );
}

#[test]
fn test_identifiers() {
    let tokens = Lexer::collect_tokens("myVar MyConverter");
    assert_eq!(
        tokens,
        vec![Token::LowerIdent("myVar"), Token::UpperIdent("MyConverter"),]
    );
}

#[test]
fn test_editor_invocation() {
    let tokens = Lexer::collect_tokens("set(x, \"value\")");
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("set"),
            Token::LParen,
            Token::LowerIdent("x"),
            Token::Comma,
            Token::StringLiteral("\"value\""),
            Token::RParen,
        ]
    );
}

#[test]
fn test_converter_invocation() {
    let tokens = Lexer::collect_tokens("Concat(a, b)[0]");
    assert_eq!(
        tokens,
        vec![
            Token::UpperIdent("Concat"),
            Token::LParen,
            Token::LowerIdent("a"),
            Token::Comma,
            Token::LowerIdent("b"),
            Token::RParen,
            Token::LBracket,
            Token::IntLiteral("0"),
            Token::RBracket,
        ]
    );
}

#[test]
fn test_path_expression() {
    let tokens = Lexer::collect_tokens("resource.attributes[\"key\"]");
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("resource"),
            Token::Dot,
            Token::LowerIdent("attributes"),
            Token::LBracket,
            Token::StringLiteral("\"key\""),
            Token::RBracket,
        ]
    );
}

#[test]
fn test_boolean_expression() {
    let tokens = Lexer::collect_tokens("x == 1 and y > 2 or not z");
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("x"),
            Token::Eq,
            Token::IntLiteral("1"),
            Token::And,
            Token::LowerIdent("y"),
            Token::Greater,
            Token::IntLiteral("2"),
            Token::Or,
            Token::Not,
            Token::LowerIdent("z"),
        ]
    );
}

#[test]
fn test_math_expression() {
    let tokens = Lexer::collect_tokens("10 + 20 * 3");
    assert_eq!(
        tokens,
        vec![
            Token::IntLiteral("10"),
            Token::Plus,
            Token::IntLiteral("20"),
            Token::Star,
            Token::IntLiteral("3"),
        ]
    );
}

#[test]
fn test_full_statement() {
    let tokens = Lexer::collect_tokens(r#"set(attributes["key"], "value") where status == 200"#);
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("set"),
            Token::LParen,
            Token::LowerIdent("attributes"),
            Token::LBracket,
            Token::StringLiteral("\"key\""),
            Token::RBracket,
            Token::Comma,
            Token::StringLiteral("\"value\""),
            Token::RParen,
            Token::Where,
            Token::LowerIdent("status"),
            Token::Eq,
            Token::IntLiteral("200"),
        ]
    );
}

#[test]
fn test_named_args() {
    let tokens = Lexer::collect_tokens("merge(target = x, source = y)");
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("merge"),
            Token::LParen,
            Token::LowerIdent("target"),
            Token::Assign,
            Token::LowerIdent("x"),
            Token::Comma,
            Token::LowerIdent("source"),
            Token::Assign,
            Token::LowerIdent("y"),
            Token::RParen,
        ]
    );
}

#[test]
fn test_map_literal() {
    let tokens = Lexer::collect_tokens(r#"{"key": "value", "count": 42}"#);
    assert_eq!(
        tokens,
        vec![
            Token::LBrace,
            Token::StringLiteral("\"key\""),
            Token::Colon,
            Token::StringLiteral("\"value\""),
            Token::Comma,
            Token::StringLiteral("\"count\""),
            Token::Colon,
            Token::IntLiteral("42"),
            Token::RBrace,
        ]
    );
}

#[test]
fn test_list_literal() {
    let tokens = Lexer::collect_tokens("[1, 2, 3]");
    assert_eq!(
        tokens,
        vec![
            Token::LBracket,
            Token::IntLiteral("1"),
            Token::Comma,
            Token::IntLiteral("2"),
            Token::Comma,
            Token::IntLiteral("3"),
            Token::RBracket,
        ]
    );
}

#[test]
fn test_enum() {
    let tokens = Lexer::collect_tokens("SPAN_KIND_SERVER STATUS_OK");
    assert_eq!(
        tokens,
        vec![
            Token::UpperIdent("SPAN_KIND_SERVER"),
            Token::UpperIdent("STATUS_OK"),
        ]
    );
}

#[test]
fn test_whitespace_handling() {
    let tokens = Lexer::collect_tokens("  set  (  x  ,  y  )  ");
    assert_eq!(
        tokens,
        vec![
            Token::LowerIdent("set"),
            Token::LParen,
            Token::LowerIdent("x"),
            Token::Comma,
            Token::LowerIdent("y"),
            Token::RParen,
        ]
    );
}

// ============================================================================
// Parser tests
// ============================================================================

use crate::Argument;

#[test]
fn test_value_equality() {
    assert_eq!(Value::Bool(true), Value::Bool(true));
    assert_eq!(Value::Int(42), Value::Int(42));
    assert_eq!(Value::Float(6.18), Value::Float(6.18));
    assert_eq!(Value::String("hello".into()), Value::String("hello".into()));
    assert_eq!(Value::Nil, Value::Nil);

    assert_ne!(Value::Bool(true), Value::Bool(false));
    assert_ne!(Value::Int(1), Value::Int(2));
    assert_ne!(Value::Bool(true), Value::Int(1));
}

#[test]
fn test_argument_access() {
    let pos = Argument::Positional(Value::Int(42));
    assert_eq!(pos.name(), None);
    assert_eq!(*pos.value(), Value::Int(42));

    let named = Argument::Named {
        name: "foo".into(),
        value: Value::String("bar".into()),
    };
    assert_eq!(named.name(), Some("foo"));
    assert_eq!(*named.value(), Value::String("bar".into()));
}

// ============================================================================
// Parser integration tests
// ============================================================================

/// Stub PathAccessor that does nothing (for testing purposes)
#[derive(Debug)]
struct StubPathAccessor;

impl PathAccessor for StubPathAccessor {
    fn get(&self, _ctx: &EvalContext, _path: &str) -> crate::Result<&Value> {
        Err("StubPathAccessor: get not implemented".into())
    }

    fn set(&self, _ctx: &mut EvalContext, _path: &str, _value: &Value) -> crate::Result<()> {
        Err("StubPathAccessor: set not implemented".into())
    }
}

/// Create a stub PathResolver that returns StubPathAccessor for any path
fn stub_path_resolver() -> PathResolver {
    Arc::new(
        |_path: &str| -> crate::Result<Arc<dyn PathAccessor + Send + Sync>> {
            Ok(Arc::new(StubPathAccessor))
        },
    )
}

/// Create a stub EvalContext
fn stub_context() -> EvalContext {
    Box::new(())
}

#[test]
fn test_parser_math_expression() {
    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let enums = EnumMap::new();
    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "-1+   2*10 - 10/5 - (1+3*2)",
    );

    // Check no parsing errors
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    // Execute and check result
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Int(10));
}

#[test]
fn test_parser_bool_expression_with_math() {
    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let enums = EnumMap::new();
    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "false or (2 < (1 + 2))",
    );

    // Check no parsing errors
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    // Execute and check result
    // false or (2 < 3) = false or true = true
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Bool(true));
}

/// PathAccessor that returns predefined values for specific paths
#[derive(Debug)]
struct MockPathAccessor {
    bool_value: Value,
    int_value: Value,
}

impl PathAccessor for MockPathAccessor {
    fn get(&self, _ctx: &EvalContext, path: &str) -> crate::Result<&Value> {
        match path {
            "my.bool.value" => Ok(&self.bool_value),
            "my.int.value" => Ok(&self.int_value),
            _ => Err(format!("Unknown path: {}", path).into()),
        }
    }

    fn set(&self, _ctx: &mut EvalContext, _path: &str, _value: &Value) -> crate::Result<()> {
        Err("MockPathAccessor: set not implemented".into())
    }
}

/// Create a PathResolver that returns MockPathAccessor with predefined values
fn mock_path_resolver(bool_value: bool, int_value: i64) -> PathResolver {
    let accessor = Arc::new(MockPathAccessor {
        bool_value: Value::Bool(bool_value),
        int_value: Value::Int(int_value),
    });
    Arc::new(
        move |_path: &str| -> crate::Result<Arc<dyn PathAccessor + Send + Sync>> {
            Ok(accessor.clone())
        },
    )
}

#[test]
fn test_parser_bool_expression_with_paths() {
    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let enums = EnumMap::new();
    // Create resolver that returns false for my.bool.value and 2 for my.int.value
    let resolver = mock_path_resolver(false, 2);
    let mut ctx = stub_context();

    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "my.bool.value or (my.int.value < (1 + 2))",
    );

    // Check no parsing errors
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    // Execute and check result
    // my.bool.value = false
    // my.int.value = 2
    // 1 + 2 = 3
    // my.int.value < 3 = 2 < 3 = true
    // false or true = true
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Bool(true));
}

#[test]
fn test_parser_math_with_converters() {
    let editors = CallbackMap::new();
    let mut converters = CallbackMap::new();
    let enums = EnumMap::new();

    // Register Sum converter: Sum(a: int, b: int) -> int { a + b }
    converters.insert(
        "Sum".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let a = match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Sum: first argument must be int".into()),
            };
            let b = match args.get(1).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Sum: second argument must be int".into()),
            };
            Ok(Value::Int(a + b))
        }),
    );

    let resolver = mock_path_resolver(false, 0);
    let mut ctx = stub_context();

    // Expression: Sum(1, 2) + 10 * Sum(-1, 1)
    // Sum(1, 2) = 1 + 2 = 3
    // Sum(-1, 1) = -1 + 1 = 0
    // 10 * 0 = 0
    // 3 + 0 = 3
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Sum(1, 2) + 10 * Sum(-1, 1)",
    );

    // Check no parsing errors
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    // Execute and check result
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Int(3));

    // Test: Sum(1,2) * Sum(2,4) = 3 * 6 = 18
    let parser2 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Sum(1,2) * Sum(2,4)",
    );

    if let Err(e) = parser2.is_error() {
        panic!("Parser error: {}", e);
    }

    let result2 = parser2.execute(&mut ctx);
    assert!(result2.is_ok(), "Execution should succeed: {:?}", result2);
    assert_eq!(result2.unwrap(), Value::Int(18));
}

#[test]
fn test_parser_math_with_enums() {
    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let mut enums = EnumMap::new();

    // Register enum values
    enums.insert("MY_INT_VALUE1".to_string(), 1);
    enums.insert("MY_INT_VALUE200".to_string(), 200);
    enums.insert("MY_INT_VALUE199".to_string(), 199);

    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    // Expression: MY_INT_VALUE200 - (MY_INT_VALUE1 + MY_INT_VALUE199)
    // 200 - (1 + 199) = 200 - 200 = 0
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "MY_INT_VALUE200 - (MY_INT_VALUE1 + MY_INT_VALUE199)",
    );

    // Check no parsing errors
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    // Execute and check result
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Int(0));

    // Test with unary minus before enum
    // Expression: -MY_INT_VALUE1 + MY_INT_VALUE200
    // -1 + 200 = 199
    let parser2 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "-MY_INT_VALUE1 + MY_INT_VALUE200",
    );

    if let Err(e) = parser2.is_error() {
        panic!("Parser error with unary minus: {}", e);
    }

    let result2 = parser2.execute(&mut ctx);
    assert!(
        result2.is_ok(),
        "Execution with unary minus should succeed: {:?}",
        result2
    );
    assert_eq!(result2.unwrap(), Value::Int(199));
}

#[test]
fn test_parser_bool_expression_with_enums() {
    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let mut enums = EnumMap::new();

    // Register enum values
    enums.insert("STATUS_OK".to_string(), 200);
    enums.insert("STATUS_NOT_FOUND".to_string(), 404);
    enums.insert("STATUS_ERROR".to_string(), 500);

    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    // Test 1: STATUS_OK < STATUS_NOT_FOUND (200 < 404 = true)
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "STATUS_OK < STATUS_NOT_FOUND",
    );

    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Bool(true));

    // Test 2: STATUS_ERROR == 500 (500 == 500 = true)
    let parser2 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "STATUS_ERROR == 500",
    );

    if let Err(e) = parser2.is_error() {
        panic!("Parser error: {}", e);
    }

    let result2 = parser2.execute(&mut ctx);
    assert!(result2.is_ok(), "Execution should succeed: {:?}", result2);
    assert_eq!(result2.unwrap(), Value::Bool(true));

    // Test 3: Complex boolean with enums
    // (STATUS_OK < STATUS_NOT_FOUND) and (STATUS_ERROR > 400)
    // (200 < 404) and (500 > 400) = true and true = true
    let parser3 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "(STATUS_OK < STATUS_NOT_FOUND) and (STATUS_ERROR > 400)",
    );

    if let Err(e) = parser3.is_error() {
        panic!("Parser error: {}", e);
    }

    let result3 = parser3.execute(&mut ctx);
    assert!(result3.is_ok(), "Execution should succeed: {:?}", result3);
    assert_eq!(result3.unwrap(), Value::Bool(true));
}

#[test]
fn test_parser_enums_as_function_args() {
    let editors = CallbackMap::new();
    let mut converters = CallbackMap::new();
    let mut enums = EnumMap::new();

    // Register enum values
    enums.insert("VALUE_10".to_string(), 10);
    enums.insert("VALUE_20".to_string(), 20);
    enums.insert("VALUE_5".to_string(), 5);

    // Register Sum converter: Sum(a: int, b: int) -> int { a + b }
    converters.insert(
        "Sum".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let a = match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Sum: first argument must be int".into()),
            };
            let b = match args.get(1).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Sum: second argument must be int".into()),
            };
            Ok(Value::Int(a + b))
        }),
    );

    // Register Multiply converter: Multiply(a: int, b: int) -> int { a * b }
    converters.insert(
        "Multiply".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let a = match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Multiply: first argument must be int".into()),
            };
            let b = match args.get(1).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v,
                _ => return Err("Multiply: second argument must be int".into()),
            };
            Ok(Value::Int(a * b))
        }),
    );

    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    // Test 1: Sum(VALUE_10, VALUE_20) + 0 = 10 + 20 + 0 = 30
    // Note: We add "+ 0" to force parsing as math expression (not bool)
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Sum(VALUE_10, VALUE_20) + 0",
    );

    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Int(30));

    // Test 2: Multiply(VALUE_5, VALUE_10) * 1 = 5 * 10 * 1 = 50
    let parser2 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Multiply(VALUE_5, VALUE_10) * 1",
    );

    if let Err(e) = parser2.is_error() {
        panic!("Parser error: {}", e);
    }

    let result2 = parser2.execute(&mut ctx);
    assert!(result2.is_ok(), "Execution should succeed: {:?}", result2);
    assert_eq!(result2.unwrap(), Value::Int(50));

    // Test 3: Nested - Sum(Multiply(VALUE_5, VALUE_10), VALUE_20) + 0 = (5*10) + 20 + 0 = 70
    let parser3 = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Sum(Multiply(VALUE_5, VALUE_10), VALUE_20) + 0",
    );

    if let Err(e) = parser3.is_error() {
        panic!("Parser error: {}", e);
    }

    let result3 = parser3.execute(&mut ctx);
    assert!(result3.is_ok(), "Execution should succeed: {:?}", result3);
    assert_eq!(result3.unwrap(), Value::Int(70));
}
