// =====================================================================================================================
//! OTTL Parser API
//!
//! This module provides a parser for the OTTL (OpenTelemetry Transformation Language)
//! that binds callbacks at parse time and produces an executable object.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Standard error type for the library
pub type BoxError = Box<dyn Error + Send + Sync>;

/// Standard result type for the library
pub type Result<T> = std::result::Result<T, BoxError>;

// =====================================================================================================================
/// User-provided context passed to callbacks during evaluation.
/// This is a placeholder type that will be refined later.
pub type EvalContext = Box<dyn Any>;

// =====================================================================================================================
/// Value Types
/// Represents all possible values in OTTL expressions and function arguments.
#[derive(Clone, Default, Debug, PartialEq /* , Eq, PartialOrd - not applicable it seems */)]
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
    fn get(&self, ctx: &EvalContext, path: &String) -> Result<&Value>;

    /// Set the value at this path in the context
    fn set(&self, ctx: &mut EvalContext, path: &String, value: &Value) -> Result<()>;
}

/// Type alias for the path resolver function.
/// Takes a path string (e.g., "body.attributes.key") and returns a PathAccessor.
pub type PathResolver =
    Arc<dyn Fn(&str) -> Result<Arc<dyn PathAccessor + Send + Sync>> + Send + Sync>;

// =====================================================================================================================
/// Callback Types

/// Callback function type for editors and converters.
/// Takes a mutable context and a list of arguments, returns a Value or error.
pub type CallbackFn = Arc<dyn Fn(&mut EvalContext, Vec<Argument>) -> Result<Value> + Send + Sync>;

/// Map of function names to their callback implementations.
pub type CallbackMap = HashMap<String, CallbackFn>;

/// Map of enum names to their integer values.
pub type EnumMap = HashMap<String, i64>;

// =====================================================================================================================
/// OTTL Parser that parses input strings and produces executable objects.
pub struct Parser<'a> {
    /// Map of editor function names to their implementations
    editors: &'a mut CallbackMap,
    /// Map of converter function names to their implementations
    converters: &'a mut CallbackMap,
    /// Map of enum names to their integer values
    enums: &'a mut EnumMap,
    /// Function to resolve paths to PathAccessor implementations
    path_resolver: &'a mut PathResolver,
}

/// Implementation of the OTTL Parser.
///
/// Provides methods for creating a parser instance and executing OTTL statements
/// against a user-provided evaluation context.
impl<'a> Parser<'a> {
    /// Creates a new parser with the given configuration.
    /// # Arguments
    ///   * `editors_map` - Map of editor function names to their callback implementations.
    ///     Editors are functions that modify the context (e.g., `set`, `delete`).
    ///   * `converters_map` - Map of converter function names to their callback implementations.
    ///     Converters are functions that transform values (e.g., `Concat`, `Int`).
    ///   * `enums_map` - Map of enum names to their integer values (e.g., `SEVERITY_INFO` -> 9).
    ///   * `path_resolver_cb` - Function to resolve path strings to PathAccessor implementations.
    ///   * `_expression` - The OTTL expression string to parse (e.g., `"set(attributes[\"key\"], \"value\")"`)
    /// # Returns a new `Parser` instance configured with the provided callbacks and ready to execute.
    pub fn new(
        editors_map: &'a mut CallbackMap,
        converters_map: &'a mut CallbackMap,
        enums_map: &'a mut EnumMap,
        path_resolver_cb: &'a mut PathResolver,
        _expression: &str, //"set(attributes[\"key\"], \"value\")"
    ) -> Self {
        Self {
            editors: editors_map,
            converters: converters_map,
            enums: enums_map,
            path_resolver: path_resolver_cb,
        }
    }

    /// Checks if the parser encountered any errors during creation (new call).
    /// # Returns Ok(()) if no errors occurred, or an error if parsing failed.
    pub fn is_error(&self) -> Result<()> {
        Ok(())
    }

    /// Executes this OTTL statement with the given context (go through AST)
    ///
    /// This method evaluates the parsed OTTL expression, invoking any bound
    /// editor or converter callbacks as needed and resolving path references.
    ///
    /// # Arguments
    ///   * `_ctx` - The mutable evaluation context that provides access to telemetry data
    ///     and can be modified by editor functions.
    /// # Returns
    ///   * `Ok(Value)` - The result of evaluating the expression, if expression has no return value => nil
    ///   * `Err(BoxError)` - An error if evaluation fails (e.g., type mismatch, missing path, callback error).
    pub fn execute(&self, _ctx: &mut EvalContext) -> Result<Value> {
        // TODO: implement actual execution logic
        Ok(Value::Nil)
    }
}
