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
        "false or not (2 < (1 + 2)) or (0xDEADBEEF == nil) or (1 != 2) or (2 >= 1.5) and (true) and \"banana 🎉\" > \"apple\"",
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
        "(((STATUS_OK < STATUS_NOT_FOUND))) and (STATUS_ERROR > 400)",
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

// ============================================================================
// Editor Statement Tests
// ============================================================================

use std::sync::Mutex;

/// Structure to capture editor call information
#[derive(Debug, Clone)]
struct EditorCallCapture {
    called: bool,
    first_arg: Option<Value>,
    second_arg: Option<Value>,
}

impl Default for EditorCallCapture {
    fn default() -> Self {
        Self {
            called: false,
            first_arg: None,
            second_arg: None,
        }
    }
}

/// PathAccessor that supports both get and set, with tracking
#[derive(Debug)]
struct TrackingPathAccessor {
    int_value: Value,
    status_code: Value,
    target_path: Value, // Dummy value for "target" path
    set_calls: Mutex<Vec<(String, Value)>>,
}

impl PathAccessor for TrackingPathAccessor {
    fn get(&self, _ctx: &EvalContext, path: &str) -> crate::Result<&Value> {
        match path {
            "my.int.value" => Ok(&self.int_value),
            "status_code" => Ok(&self.status_code),
            "target" | "x" => Ok(&self.target_path), // For editor's first argument
            _ => Err(format!("Unknown path: {}", path).into()),
        }
    }

    fn set(&self, _ctx: &mut EvalContext, path: &str, value: &Value) -> crate::Result<()> {
        self.set_calls
            .lock()
            .unwrap()
            .push((path.to_string(), value.clone()));
        Ok(())
    }
}

/// Create a PathResolver with tracking accessor
fn tracking_path_resolver(
    int_value: i64,
    status_code: i64,
) -> (PathResolver, Arc<TrackingPathAccessor>) {
    let accessor = Arc::new(TrackingPathAccessor {
        int_value: Value::Int(int_value),
        status_code: Value::Int(status_code),
        target_path: Value::Nil, // Dummy value for target path
        set_calls: Mutex::new(Vec::new()),
    });
    let accessor_clone = accessor.clone();
    let resolver: PathResolver = Arc::new(
        move |_path: &str| -> crate::Result<Arc<dyn PathAccessor + Send + Sync>> {
            Ok(accessor_clone.clone())
        },
    );
    (resolver, accessor)
}

#[test]
fn test_editor_executes_when_condition_true() {
    // Setup: condition will be TRUE
    // my.int.value = 50 (> 0)
    // status_code = 200 (== STATUS_OK)

    let call_capture = Arc::new(Mutex::new(EditorCallCapture::default()));
    let capture_clone = call_capture.clone();

    let mut editors = CallbackMap::new();
    editors.insert(
        "set".to_string(),
        Arc::new(move |_ctx: &mut EvalContext, args: Vec<Argument>| {
            let mut capture = capture_clone.lock().unwrap();
            capture.called = true;
            capture.first_arg = args.get(0).map(|a| a.value().clone());
            capture.second_arg = args.get(1).map(|a| a.value().clone());
            Ok(Value::Nil)
        }),
    );

    let mut converters = CallbackMap::new();
    // Sum converter: Sum(a, b) -> a + b
    converters.insert(
        "Sum".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let a = match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v as f64,
                Some(Value::Float(v)) => *v,
                _ => return Err("Sum: first argument must be numeric".into()),
            };
            let b = match args.get(1).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v as f64,
                Some(Value::Float(v)) => *v,
                _ => return Err("Sum: second argument must be numeric".into()),
            };
            Ok(Value::Float(a + b))
        }),
    );

    let mut enums = EnumMap::new();
    enums.insert("STATUS_WEIGHT".to_string(), 100);
    enums.insert("STATUS_OK".to_string(), 200);

    // my.int.value = 50, status_code = 200 (matches STATUS_OK)
    let (resolver, _accessor) = tracking_path_resolver(50, 200);
    let mut ctx = stub_context();

    // Expression:
    // set(target, Sum(STATUS_WEIGHT, my.int.value) * 1.5) where my.int.value > 0 and status_code == STATUS_OK
    // Sum(100, 50) * 1.5 = 150 * 1.5 = 225.0
    // Condition: 50 > 0 (true) and 200 == 200 (true) -> true
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "set(target, Sum(STATUS_WEIGHT, my.int.value) * 1.5) where my.int.value > 0 and status_code == STATUS_OK",
    );

    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Nil); // Editor returns Nil

    // Verify editor was called
    let capture = call_capture.lock().unwrap();
    assert!(capture.called, "Editor 'set' should have been called");

    // Verify first argument (path "target" resolves to Nil in our mock)
    assert_eq!(
        capture.first_arg,
        Some(Value::Nil),
        "First argument should be the resolved path value (Nil)"
    );

    // Verify second argument (computed value = 225.0)
    // Sum(STATUS_WEIGHT=100, my.int.value=50) * 1.5 = 150 * 1.5 = 225.0
    assert_eq!(
        capture.second_arg,
        Some(Value::Float(225.0)),
        "Second argument should be Sum(100, 50) * 1.5 = 225.0"
    );
}

#[test]
fn test_editor_not_executed_when_condition_false() {
    // Setup: condition will be FALSE
    // my.int.value = -10 (NOT > 0)
    // status_code = 200 (== STATUS_OK, but first part is false)

    let call_capture = Arc::new(Mutex::new(EditorCallCapture::default()));
    let capture_clone = call_capture.clone();

    let mut editors = CallbackMap::new();
    editors.insert(
        "set".to_string(),
        Arc::new(move |_ctx: &mut EvalContext, args: Vec<Argument>| {
            let mut capture = capture_clone.lock().unwrap();
            capture.called = true;
            capture.first_arg = args.get(0).map(|a| a.value().clone());
            capture.second_arg = args.get(1).map(|a| a.value().clone());
            Ok(Value::Nil)
        }),
    );

    let mut converters = CallbackMap::new();
    converters.insert(
        "Sum".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let a = match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v as f64,
                Some(Value::Float(v)) => *v,
                _ => return Err("Sum: first argument must be numeric".into()),
            };
            let b = match args.get(1).map(|arg| arg.value()) {
                Some(Value::Int(v)) => *v as f64,
                Some(Value::Float(v)) => *v,
                _ => return Err("Sum: second argument must be numeric".into()),
            };
            Ok(Value::Float(a + b))
        }),
    );

    let mut enums = EnumMap::new();
    enums.insert("STATUS_WEIGHT".to_string(), 100);
    enums.insert("STATUS_OK".to_string(), 200);

    // my.int.value = -10 (negative!), status_code = 200
    let (resolver, _accessor) = tracking_path_resolver(-10, 200);
    let mut ctx = stub_context();

    // Expression:
    // set(target, Sum(STATUS_WEIGHT, my.int.value) * 1.5) where my.int.value > 0 and status_code == STATUS_OK
    // Condition: -10 > 0 (FALSE) and 200 == 200 (true) -> false (short-circuit)
    // Editor should NOT be called
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "set(target, Sum(STATUS_WEIGHT, my.int.value) * 1.5) where my.int.value > 0 and status_code == STATUS_OK",
    );

    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Nil); // Still returns Nil

    // Verify editor was NOT called
    let capture = call_capture.lock().unwrap();
    assert!(
        !capture.called,
        "Editor 'set' should NOT have been called when condition is false"
    );
    assert!(
        capture.first_arg.is_none(),
        "No arguments should be captured"
    );
    assert!(
        capture.second_arg.is_none(),
        "No arguments should be captured"
    );
}

#[test]
fn test_editor_set_list_of_maps() {
    // Test: set(x, [{"id": 1, "value": Double(5.0)}, {"id": 2, "value": STATUS_OK}, {"id": 3, "value": my.int.value}])
    // Expected result: x = [{"id": 1, "value": 10.0}, {"id": 2, "value": 200}, {"id": 3, "value": 73}]

    let call_capture = Arc::new(Mutex::new(EditorCallCapture::default()));
    let capture_clone = call_capture.clone();

    let mut editors = CallbackMap::new();
    editors.insert(
        "set".to_string(),
        Arc::new(move |_ctx: &mut EvalContext, args: Vec<Argument>| {
            let mut capture = capture_clone.lock().unwrap();
            capture.called = true;
            capture.first_arg = args.get(0).map(|a| a.value().clone());
            capture.second_arg = args.get(1).map(|a| a.value().clone());
            Ok(Value::Nil)
        }),
    );

    let mut converters = CallbackMap::new();
    // Double converter: Double(x) -> x * 2
    converters.insert(
        "Double".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            match args.get(0).map(|arg| arg.value()) {
                Some(Value::Int(v)) => Ok(Value::Int(v * 2)),
                Some(Value::Float(v)) => Ok(Value::Float(v * 2.0)),
                _ => Err("Double: argument must be numeric".into()),
            }
        }),
    );

    let mut enums = EnumMap::new();
    enums.insert("STATUS_OK".to_string(), 200);

    // my.int.value = 73
    let (resolver, _accessor) = tracking_path_resolver(73, 200);
    let mut ctx = stub_context();

    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "set(x, [{\"id\": 1, \"value\": Double(5.0) ,}, {\"id\": 2, \"value\": STATUS_OK}, {\"id\": 3, \"value\": my.int.value}])",
    );

    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }

    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Nil);

    // Verify editor was called
    let capture = call_capture.lock().unwrap();
    assert!(capture.called, "Editor 'set' should have been called");

    // Verify first argument (path "x" resolves to Nil in our mock)
    assert_eq!(
        capture.first_arg,
        Some(Value::Nil),
        "First argument should be the resolved path value"
    );

    // Verify second argument - list of maps with computed values
    // Expected: [{"id": 1, "value": 10.0}, {"id": 2, "value": 200}, {"id": 3, "value": 73}]
    use std::collections::HashMap;

    let mut map1 = HashMap::new();
    map1.insert("id".to_string(), Value::Int(1));
    map1.insert("value".to_string(), Value::Float(10.0));

    let mut map2 = HashMap::new();
    map2.insert("id".to_string(), Value::Int(2));
    map2.insert("value".to_string(), Value::Int(200));

    let mut map3 = HashMap::new();
    map3.insert("id".to_string(), Value::Int(3));
    map3.insert("value".to_string(), Value::Int(73));

    let expected_list = Value::List(vec![Value::Map(map1), Value::Map(map2), Value::Map(map3)]);

    assert_eq!(
        capture.second_arg,
        Some(expected_list),
        "Second argument should be the list of maps with computed values"
    );
}

// ============================================================================
// Path Expressions Tests
// ============================================================================

/// PathAccessor that supports multi-level paths, index access, and key access
#[derive(Debug)]
struct PathExprAccessor {
    // resource.attributes.status = 200
    // resource.count = 10
    resource_status: Value,
    resource_count: Value,
    // items[0] = 5, items[1] = 3
    items: Value,
    // data["key"] = 100, data["multiplier"] = 2
    data: Value,
}

impl PathAccessor for PathExprAccessor {
    fn get(&self, _ctx: &EvalContext, path: &str) -> crate::Result<&Value> {
        match path {
            "resource.attributes.status" => Ok(&self.resource_status),
            "resource.count" => Ok(&self.resource_count),
            "items" => Ok(&self.items),
            "data" => Ok(&self.data),
            _ => Err(format!("Unknown path: {}", path).into()),
        }
    }

    fn set(&self, _ctx: &mut EvalContext, _path: &str, _value: &Value) -> crate::Result<()> {
        Err("PathExprAccessor: set not implemented".into())
    }
}

#[test]
fn test_parser_path_expressions_comprehensive() {
    // This test verifies all path expression types in one boolean expression:
    // - Multi-level paths: resource.attributes.status
    // - Index access by number: items[0], items[1]
    // - Index access by key: data["key"]
    // - Math expressions with paths: items[0] + items[1]

    let editors = CallbackMap::new();
    let converters = CallbackMap::new();
    let enums = EnumMap::new();

    use std::collections::HashMap;

    // Setup data map: {"key": 100, "multiplier": 2}
    let mut data_map = HashMap::new();
    data_map.insert("key".to_string(), Value::Int(100));
    data_map.insert("multiplier".to_string(), Value::Int(2));

    let accessor = Arc::new(PathExprAccessor {
        resource_status: Value::Int(200),
        resource_count: Value::Int(10),
        items: Value::List(vec![Value::Int(5), Value::Int(3)]),
        data: Value::Map(data_map),
    });
    let accessor_clone = accessor.clone();

    let resolver: PathResolver = Arc::new(
        move |_path: &str| -> crate::Result<Arc<dyn PathAccessor + Send + Sync>> {
            Ok(accessor_clone.clone())
        },
    );

    let mut ctx = stub_context();

    // Combined expression testing all path types:
    // (resource.attributes.status == 200) and (items[0] + items[1] == 8) and (data["key"] == 100)
    // - resource.attributes.status = 200 → true
    // - items[0] + items[1] = 5 + 3 = 8 → true
    // - data["key"] = 100 → true
    // Result: true and true and true = true
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "(resource.attributes.status == 200) and (items[0] + items[1] == 8) and (data[\"key\"] == 100)",
    );
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution failed: {:?}", result);
    assert_eq!(
        result.unwrap(),
        Value::Bool(true),
        "Combined path expression should be true"
    );
}

// ============================================================================
// Converter with Index Tests
// ============================================================================

#[test]
fn test_converter_with_index() {
    // Test: Split("a,b,c", ",")[0] should return "a"
    let editors = CallbackMap::new();
    let mut converters = CallbackMap::new();
    let enums = EnumMap::new();

    // Split converter: splits string by delimiter, returns list
    converters.insert(
        "Split".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            let text = match args.get(0).map(|arg| arg.value()) {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("Split first argument must be string".into()),
            };
            let delimiter = match args.get(1).map(|arg| arg.value()) {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("Split second argument must be string".into()),
            };
            let parts: Vec<Value> = text
                .split(&delimiter)
                .map(|s| Value::String(s.to_string()))
                .collect();
            Ok(Value::List(parts))
        }),
    );

    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    // Split("a,b,c", ",")[0] == "a"
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Split(\"a,b,c\", \",\")[0] == \"a\"",
    );
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution failed: {:?}", result);
    assert_eq!(
        result.unwrap(),
        Value::Bool(true),
        "Split(\"a,b,c\", \",\")[0] should equal \"a\""
    );
}

// ============================================================================
// Named Arguments Tests
// ============================================================================

#[test]
fn test_named_arguments() {
    // Test: Convert(value=10, format="hex") with named arguments
    let editors = CallbackMap::new();
    let mut converters = CallbackMap::new();
    let enums = EnumMap::new();

    // Convert converter: converts value based on format
    // Uses named arguments: value and format
    converters.insert(
        "Convert".to_string(),
        Arc::new(|_ctx: &mut EvalContext, args: Vec<Argument>| {
            // Find arguments by name
            let value_arg = args
                .iter()
                .find(|a| a.name().as_deref() == Some("value"))
                .or_else(|| args.get(0))
                .ok_or("Convert requires value argument")?;
            let format_arg = args
                .iter()
                .find(|a| a.name().as_deref() == Some("format"))
                .or_else(|| args.get(1))
                .ok_or("Convert requires format argument")?;

            let value = match value_arg.value() {
                Value::Int(n) => n,
                _ => return Err("value must be integer".into()),
            };
            let format = match format_arg.value() {
                Value::String(s) => s.clone(),
                _ => return Err("format must be string".into()),
            };

            match format.as_str() {
                "hex" => Ok(Value::String(format!("{:x}", value))),
                "binary" => Ok(Value::String(format!("{:b}", value))),
                "octal" => Ok(Value::String(format!("{:o}", value))),
                _ => Ok(Value::String(value.to_string())),
            }
        }),
    );

    let resolver = stub_path_resolver();
    let mut ctx = stub_context();

    // Convert(value=10, format="hex") == "a"
    let parser = Parser::new(
        &editors,
        &converters,
        &enums,
        &resolver,
        "Convert(value=10, format=\"hex\") == \"a\"",
    );
    if let Err(e) = parser.is_error() {
        panic!("Parser error: {}", e);
    }
    let result = parser.execute(&mut ctx);
    assert!(result.is_ok(), "Execution failed: {:?}", result);
    assert_eq!(
        result.unwrap(),
        Value::Bool(true),
        "Convert(value=10, format=\"hex\") should equal \"a\""
    );
}
