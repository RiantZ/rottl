// =====================================================================================================================
//! OTTL Parser API
//!
//! This module provides a parser for the OTTL (OpenTelemetry Transformation Language)
//! that binds callbacks at parse time and produces an executable object.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// =====================================================================================================================
/// User-provided context passed to callbacks during evaluation.
/// This is a placeholder type that will be refined later.
pub type Context = Box<dyn Any>;

// =====================================================================================================================
/// Value Types
/// Represents all possible values in OTTL expressions and function arguments.
#[derive(Clone, Default)]
pub enum Value {
    /// Boolean value (true/false)
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// String value
    /// AZH: TODO: condier to use Arc for reference counting in case of clonning and not making full copy with going to heap, locks, etc.
    String(String),
    /// Bytes literal (e.g., 0xDEADBEEF)
    Bytes(Vec<u8>),
    /// Nil/null value
    #[default] // set nil default for all below!!! ;)
    Nil,
    /// List of values
    List(Vec<Value>),
    /// Map of string keys to values
    Map(HashMap<String, Value>),
    /// A callable function (editor or converter bound at parse time)
    Function(CallbackFn),
    /// A path accessor (can read and write)
    Path(Arc<dyn PathAccessor + Send + Sync>),
    /// A boolean expression (lazily evaluated)
    BooleanExpr(Arc<dyn Fn(&mut Context) -> Result<bool, EvalError> + Send + Sync>),
    /// A math expression (lazily evaluated)
    MathExpr(Arc<dyn Fn(&mut Context) -> EvalResult + Send + Sync>),
}

// =====================================================================================================================
/// Custom [`Debug`] implementation for [`Value`].
///
/// This implementation is required because some variants contain types that don't
/// implement `Debug` (e.g., `Arc<dyn Fn...>`). For such variants (`Function`,
/// `BooleanExpr`, `MathExpr`), a placeholder `<fn>` is displayed instead of the
/// actual content.
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(v) => f.debug_tuple("Bool").field(v).finish(),
            Value::Int(v) => f.debug_tuple("Int").field(v).finish(),
            Value::Float(v) => f.debug_tuple("Float").field(v).finish(),
            Value::String(v) => f.debug_tuple("String").field(v).finish(),
            Value::Bytes(v) => f.debug_tuple("Bytes").field(v).finish(),
            Value::Nil => write!(f, "Nil"),
            Value::List(v) => f.debug_tuple("List").field(v).finish(),
            Value::Map(v) => f.debug_tuple("Map").field(v).finish(),
            Value::Function(func) => write!(f, "Function({:p})", Arc::as_ptr(func)),
            Value::Path(p) => f.debug_tuple("Path").field(p).finish(),
            Value::BooleanExpr(bool_exp) => write!(f, "BooleanExpr({:p})", Arc::as_ptr(bool_exp)),
            Value::MathExpr(math_exp) => write!(f, "MathExpr{:p})", Arc::as_ptr(math_exp)),
        }
    }
}

// =====================================================================================================================
/// Custom [`PartialEq`] implementation for [`Value`].
///
/// This implementation is required because some variants contain types that don't
/// implement `PartialEq` (e.g., `Arc<dyn Fn...>`, `Arc<dyn PathAccessor...>`).
///
/// # Equality semantics
///
/// - Primitive types (`Bool`, `Int`, `Float`, `String`, `Bytes`, `Nil`) are compared by value.
/// - `List` and `Map` are compared recursively by their contents.
/// - `Function`, `Path`, `BooleanExpr`, and `MathExpr` variants are **never equal**
///   to any value (including themselves), as closures and trait objects cannot be
///   meaningfully compared.
///
/// # Note
///
/// This implementation does **not** satisfy reflexivity for function-like variants
/// (`Function`, `BooleanExpr`, `MathExpr`, `Path`), meaning `x == x` returns `false`
/// for these types. Use with caution in contexts that assume full equivalence.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            (Value::Nil, Value::Nil) => true,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            // Functions, Paths, and expressions cannot be compared for equality
            _ => false,
        }
    }
}

// =====================================================================================================================
/// Argument passed to callback functions.
/// Can be either a positional argument or a named argument.
#[derive(Debug, Clone)]
pub enum Argument {
    /// Positional argument with just a value
    Positional(Value),
    /// Named argument with name and value
    Named { name: String, value: Value },
}

/// Methods for extracting data from [`Argument`] values.
///
/// These methods provide a unified interface for accessing argument values
/// regardless of whether the argument is positional or named.
///
/// # Examples
///
/// ```ignore
/// let pos_arg = Argument::Positional(Value::Int(42));
/// let named_arg = Argument::Named { name: String::from("count"), value: Value::Int(10) };
///
/// Both return the inner value
/// assert_eq!(pos_arg.value(), &Value::Int(42));
/// assert_eq!(named_arg.value(), &Value::Int(10));
///
/// Only named arguments have a name
/// assert_eq!(pos_arg.name(), None);
/// assert_eq!(named_arg.name(), Some("count"));
/// ```
impl Argument {
    /// Returns a reference to the value of this argument.
    pub fn value(&self) -> &Value {
        match self {
            Argument::Positional(v) => v,
            Argument::Named { value, .. } => value,
        }
    }

    /// Get the value, consuming this argument
    /// AZH: not sure yet that we need that
    //pub fn into_value(self) -> Value {
    //    match self {
    //        Argument::Positional(v) => v,
    //        Argument::Named { value, .. } => value,
    //    }
    //}

    /// Get the name if this is a named argument
    pub fn name(&self) -> Option<&str> {
        match self {
            Argument::Positional(_) => None,
            Argument::Named { name, .. } => Some(name),
        }
    }
}

// =====================================================================================================================
/// Trait for accessing (reading and writing) path values in the context.
pub trait PathAccessor: fmt::Debug {
    /// Get the value at this path from the context
    fn get(&self, ctx: &Context, path: Vec<String>) -> Result<Value, EvalError>;

    /// Set the value at this path in the context
    fn set(&self, ctx: &mut Context, path: Vec<String>, value: Value) -> Result<(), EvalError>;
}

/// Type alias for the path resolver function.
/// Takes a path string (e.g., "body.attributes.key") and returns a PathAccessor.
pub type PathResolver =
    Arc<dyn Fn(&str) -> Result<Arc<dyn PathAccessor + Send + Sync>, ParseError> + Send + Sync>;

// =====================================================================================================================
/// Callback Types

/// Result type for evaluation operations
pub type EvalResult = Result<Value, EvalError>;

/// Callback function type for editors and converters.
/// Takes a mutable context and a list of arguments, returns a Value or error.
pub type CallbackFn = Arc<dyn Fn(&mut Context, Vec<Argument>) -> EvalResult + Send + Sync>;

/// Map of function names to their callback implementations.
pub type CallbackMap = HashMap<String, CallbackFn>;

/// Map of enum names to their integer values.
pub type EnumMap = HashMap<String, i64>;

// ============================================================================
// Errors
// ============================================================================

/// Error that occurs during parsing.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Position in the input where the error occurred (byte offset)
    pub position: Option<usize>,
    /// Expected tokens or constructs
    pub expected: Vec<String>,
    /// What was actually found
    pub found: Option<String>,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position: None,
            expected: Vec::new(),
            found: None,
        }
    }

    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    pub fn with_expected(mut self, expected: Vec<String>) -> Self {
        self.expected = expected;
        self
    }

    pub fn with_found(mut self, found: impl Into<String>) -> Self {
        self.found = Some(found.into());
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error: {}", self.message)?;
        if let Some(pos) = self.position {
            write!(f, " at position {}", pos)?;
        }
        if !self.expected.is_empty() {
            write!(f, ", expected: {}", self.expected.join(", "))?;
        }
        if let Some(ref found) = self.found {
            write!(f, ", found: {}", found)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// Error that occurs during evaluation/execution.
#[derive(Debug, Clone)]
pub struct EvalError {
    /// Error message
    pub message: String,
    /// Error kind for programmatic handling
    pub kind: EvalErrorKind,
}

/// Kinds of evaluation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalErrorKind {
    /// Type mismatch (e.g., adding string to int)
    TypeMismatch,
    /// Division by zero
    DivisionByZero,
    /// Index out of bounds
    IndexOutOfBounds,
    /// Key not found in map
    KeyNotFound,
    /// Path not found or invalid
    InvalidPath,
    /// Function not found
    FunctionNotFound,
    /// Invalid argument count or types
    InvalidArguments,
    /// Nil value where not expected
    UnexpectedNil,
    /// Generic runtime error
    Runtime,
}

impl EvalError {
    pub fn new(kind: EvalErrorKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    pub fn type_mismatch(message: impl Into<String>) -> Self {
        Self::new(EvalErrorKind::TypeMismatch, message)
    }

    pub fn division_by_zero() -> Self {
        Self::new(EvalErrorKind::DivisionByZero, "Division by zero")
    }

    pub fn index_out_of_bounds(index: i64, len: usize) -> Self {
        Self::new(
            EvalErrorKind::IndexOutOfBounds,
            format!("Index {} out of bounds for length {}", index, len),
        )
    }

    pub fn key_not_found(key: &str) -> Self {
        Self::new(
            EvalErrorKind::KeyNotFound,
            format!("Key '{}' not found", key),
        )
    }

    pub fn invalid_path(path: &str) -> Self {
        Self::new(
            EvalErrorKind::InvalidPath,
            format!("Invalid path: {}", path),
        )
    }

    pub fn function_not_found(name: &str) -> Self {
        Self::new(
            EvalErrorKind::FunctionNotFound,
            format!("Function '{}' not found", name),
        )
    }

    pub fn invalid_arguments(message: impl Into<String>) -> Self {
        Self::new(EvalErrorKind::InvalidArguments, message)
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(EvalErrorKind::Runtime, message)
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Evaluation error ({:?}): {}", self.kind, self.message)
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Parser Configuration
// ============================================================================

/// Configuration for the OTTL parser.
/// Contains all the callbacks and resolvers needed to parse and bind an OTTL statement.
pub struct ParserConfig {
    /// Map of editor function names to their implementations
    pub editors: CallbackMap,
    /// Map of converter function names to their implementations
    pub converters: CallbackMap,
    /// Map of enum names to their integer values
    pub enums: EnumMap,
    /// Function to resolve paths to PathAccessor implementations
    pub path_resolver: PathResolver,
}

impl ParserConfig {
    /// Create a new parser configuration with empty maps and a default path resolver.
    pub fn new() -> Self {
        Self {
            editors: HashMap::new(),
            converters: HashMap::new(),
            enums: HashMap::new(),
            path_resolver: Arc::new(|path| {
                Err(ParseError::new(format!(
                    "No path resolver configured for path: {}",
                    path
                )))
            }),
        }
    }

    /// Set the editors map
    pub fn with_editors(mut self, editors: CallbackMap) -> Self {
        self.editors = editors;
        self
    }

    /// Set the converters map
    pub fn with_converters(mut self, converters: CallbackMap) -> Self {
        self.converters = converters;
        self
    }

    /// Set the enums map
    pub fn with_enums(mut self, enums: EnumMap) -> Self {
        self.enums = enums;
        self
    }

    /// Set the path resolver
    pub fn with_path_resolver(mut self, resolver: PathResolver) -> Self {
        self.path_resolver = resolver;
        self
    }

    /// Register an editor function
    pub fn register_editor(&mut self, name: impl Into<String>, callback: CallbackFn) {
        self.editors.insert(name.into(), callback);
    }

    /// Register a converter function
    pub fn register_converter(&mut self, name: impl Into<String>, callback: CallbackFn) {
        self.converters.insert(name.into(), callback);
    }

    /// Register an enum value
    pub fn register_enum(&mut self, name: impl Into<String>, value: i64) {
        self.enums.insert(name.into(), value);
    }
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Executable
// ============================================================================

/// An executable OTTL statement that has been parsed and bound to callbacks.
/// Call `execute` with a context to run the statement.
pub struct Executable {
    /// The bound execution function
    execute_fn: Box<dyn Fn(&mut Context) -> EvalResult + Send + Sync>,
    /// Original source for debugging
    source: String,
}

impl Executable {
    /// Create a new executable from a bound function
    pub(crate) fn new(
        execute_fn: Box<dyn Fn(&mut Context) -> EvalResult + Send + Sync>,
        source: String,
    ) -> Self {
        Self { execute_fn, source }
    }

    /// Execute this OTTL statement with the given context.
    /// Returns the result value or an evaluation error.
    pub fn execute(&self, ctx: &mut Context) -> EvalResult {
        (self.execute_fn)(ctx)
    }

    /// Get the original source string
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl fmt::Debug for Executable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executable")
            .field("source", &self.source)
            .finish()
    }
}

// ============================================================================
// Parser
// ============================================================================

/// OTTL Parser that parses input strings and produces executable objects.
pub struct Parser {
    config: ParserConfig,
}

impl Parser {
    /// Create a new parser with the given configuration
    pub fn new(config: ParserConfig) -> Self {
        Self { config }
    }

    /// Parse an OTTL statement and return an executable object.
    ///
    /// The returned `Executable` has all callbacks bound and can be executed
    /// multiple times with different contexts.
    ///
    /// # Arguments
    /// * `input` - The OTTL statement to parse
    ///
    /// # Returns
    /// * `Ok(Executable)` - Successfully parsed and bound executable
    /// * `Err(ParseError)` - Parse error with details
    ///
    /// # Example
    /// ```ignore
    /// let config = ParserConfig::new()
    ///     .with_editors(editors)
    ///     .with_converters(converters)
    ///     .with_enums(enums)
    ///     .with_path_resolver(resolver);
    ///
    /// let parser = Parser::new(config);
    /// let executable = parser.parse("set(attributes[\"key\"], \"value\")")?;
    ///
    /// let mut ctx: Context = Box::new(MyContext::new());
    /// let result = executable.execute(&mut ctx)?;
    /// ```
    pub fn parse(&self, input: &str) -> Result<Executable, ParseError> {
        // TODO: Implement actual parsing using chumsky
        // This is a placeholder that will be implemented
        let _ = &self.config;
        Err(ParseError::new(format!(
            "Parser not yet implemented for input: {}",
            input
        )))
    }

    /// Get a reference to the parser configuration
    pub fn config(&self) -> &ParserConfig {
        &self.config
    }

    /// Get a mutable reference to the parser configuration
    pub fn config_mut(&mut self) -> &mut ParserConfig {
        &mut self.config
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Parse an OTTL statement with the given configuration.
/// This is a convenience function that creates a parser and parses the input.
pub fn parse(input: &str, config: ParserConfig) -> Result<Executable, ParseError> {
    Parser::new(config).parse(input)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_equality() {
        assert_eq!(Value::Bool(true), Value::Bool(true));
        assert_eq!(Value::Int(42), Value::Int(42));
        assert_eq!(Value::Float(3.14), Value::Float(3.14));
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

    #[test]
    fn test_parser_config_builder() {
        let config = ParserConfig::new().with_enums({
            let mut m = HashMap::new();
            m.insert("SEVERITY_INFO".into(), 9);
            m
        });

        assert_eq!(config.enums.get("SEVERITY_INFO"), Some(&9));
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::new("Unexpected token")
            .with_position(42)
            .with_expected(vec!["identifier".into(), "number".into()])
            .with_found("')'");

        let msg = err.to_string();
        assert!(msg.contains("Unexpected token"));
        assert!(msg.contains("42"));
        assert!(msg.contains("identifier"));
        assert!(msg.contains("')'"));
    }

    #[test]
    fn test_eval_error_constructors() {
        let err = EvalError::type_mismatch("Cannot add string to int");
        assert_eq!(err.kind, EvalErrorKind::TypeMismatch);

        let err = EvalError::division_by_zero();
        assert_eq!(err.kind, EvalErrorKind::DivisionByZero);

        let err = EvalError::index_out_of_bounds(5, 3);
        assert_eq!(err.kind, EvalErrorKind::IndexOutOfBounds);
        assert!(err.message.contains("5"));
        assert!(err.message.contains("3"));
    }
}
