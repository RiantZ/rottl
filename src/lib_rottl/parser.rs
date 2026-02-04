//! OTTL Parser Implementation
//!
//! This module contains the Parser struct and its implementation using chumsky.

use chumsky::prelude::*;
use chumsky::Parser as ChumskyParser;

use super::lexer::Token;
use super::{
    Argument, BoxError, CallbackFn, CallbackMap, EnumMap, EvalContext, OttlParser, PathResolver,
    Result, Value,
};

// =====================================================================================================================
// AST Node Types
// =====================================================================================================================

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum CompOp {
    Eq,
    NotEq,
    Less,
    Greater,
    LessEq,
    GreaterEq,
}

/// Index for path or converter result
#[derive(Debug, Clone)]
pub enum IndexExpr {
    /// String index like ["key"]
    String(String),
    /// Integer index like [0]
    Int(i64),
}

/// Path expression (e.g., resource.attributes["key"])
#[derive(Debug, Clone)]
pub struct PathExpr {
    /// Segments of the path (e.g., ["resource", "attributes"])
    pub segments: Vec<String>,
    /// Optional indexes
    pub indexes: Vec<IndexExpr>,
}

/// Function invocation (Editor or Converter)
#[derive(Clone)]
pub struct FunctionCall {
    /// Function name
    pub name: String,
    /// Whether this is an editor (lowercase) or converter (uppercase)
    pub is_editor: bool,
    /// Arguments
    pub args: Vec<ArgExpr>,
    /// Optional indexes (for converters)
    pub indexes: Vec<IndexExpr>,
    /// Callback reference (resolved at parse time)
    pub callback: Option<CallbackFn>,
}

impl std::fmt::Debug for FunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCall")
            .field("name", &self.name)
            .field("is_editor", &self.is_editor)
            .field("args", &self.args)
            .field("indexes", &self.indexes)
            .field("callback", &self.callback.is_some())
            .finish()
    }
}

/// Argument expression
#[derive(Debug, Clone)]
pub enum ArgExpr {
    /// Positional argument
    Positional(ValueExpr),
    /// Named argument
    Named { name: String, value: ValueExpr },
}

/// Value expression - any value in OTTL
#[derive(Debug, Clone)]
pub enum ValueExpr {
    /// Literal value
    Literal(Value),
    /// Path expression
    Path(PathExpr),
    /// List literal
    List(Vec<ValueExpr>),
    /// Map literal
    Map(Vec<(String, ValueExpr)>),
    /// Function call (converter or editor)
    FunctionCall(Box<FunctionCall>),
    /// Math expression
    Math(Box<MathExpr>),
}

/// Combined math operator
#[derive(Debug, Clone)]
pub enum MathOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Math expression with operator precedence
#[derive(Debug, Clone)]
pub enum MathExpr {
    /// Primary value (literal, path, converter, or grouped expression)
    Primary(ValueExpr),
    /// Unary negation
    Negate(Box<MathExpr>),
    /// Binary operation: term (+/-) or factor (*/)
    Binary {
        left: Box<MathExpr>,
        op: MathOp,
        right: Box<MathExpr>,
    },
}

/// Boolean expression with operator precedence
#[derive(Debug, Clone)]
pub enum BoolExpr {
    /// Literal boolean
    Literal(bool),
    /// Comparison expression
    Comparison {
        left: ValueExpr,
        op: CompOp,
        right: ValueExpr,
    },
    /// Converter call returning boolean
    Converter(Box<FunctionCall>),
    /// Path that evaluates to boolean
    Path(PathExpr),
    /// Logical NOT
    Not(Box<BoolExpr>),
    /// Logical AND
    And(Box<BoolExpr>, Box<BoolExpr>),
    /// Logical OR
    Or(Box<BoolExpr>, Box<BoolExpr>),
}

/// Editor invocation statement
#[derive(Debug, Clone)]
pub struct EditorStatement {
    /// The editor function call
    pub editor: FunctionCall,
    /// Optional WHERE clause condition
    pub condition: Option<BoolExpr>,
}

/// Root AST node - either an editor statement, a boolean expression, or a math expression
#[derive(Debug, Clone)]
pub enum RootExpr {
    /// Editor invocation with optional WHERE clause
    EditorStatement(EditorStatement),
    /// Standalone boolean expression
    BooleanExpression(BoolExpr),
    /// Standalone math expression
    MathExpression(MathExpr),
}

// =====================================================================================================================
// Parser Implementation
// =====================================================================================================================

/// OTTL Parser that parses input strings and produces executable objects.
///
/// Create a parser using [`Parser::new`], then use the [`OttlParser`] trait methods
/// to check for errors and execute the statement.
pub struct Parser {
    /// Parsed AST
    ast: Option<RootExpr>,
    /// Parsing errors
    errors: Vec<String>,
    /// Path resolver for reading/writing paths
    path_resolver: PathResolver,
}

impl Parser {
    /// Creates a new parser with the given configuration.
    ///
    /// # Arguments
    ///   * `editors_map` - Map of editor function names to their callback implementations.
    ///     Editors are functions that modify the context (e.g., `set`, `delete`).
    ///   * `converters_map` - Map of converter function names to their callback implementations.
    ///     Converters are functions that transform values (e.g., `Concat`, `Int`).
    ///   * `enums_map` - Map of enum names to their integer values (e.g., `SEVERITY_INFO` -> 9).
    ///   * `path_resolver_cb` - Function to resolve path strings to PathAccessor implementations.
    ///   * `expression` - The OTTL expression string to parse (e.g., `"set(attributes[\"key\"], \"value\")"`)
    ///
    /// # Returns
    ///
    /// A new `Parser` instance configured with the provided callbacks and ready to execute.
    pub fn new(
        editors_map: &mut CallbackMap,
        converters_map: &mut CallbackMap,
        enums_map: &mut EnumMap,
        path_resolver_cb: &mut PathResolver,
        expression: &str,
    ) -> Self {
        let mut parser = Parser {
            ast: None,
            errors: Vec::new(),
            path_resolver: path_resolver_cb.clone(),
        };

        // Tokenize the input
        let tokens = super::lexer::Lexer::collect_with_spans(expression);

        if tokens.is_empty() && !expression.trim().is_empty() {
            parser.errors.push("Lexer failed to tokenize input".into());
            return parser;
        }

        // Build the chumsky parser
        let chumsky_parser = build_parser(editors_map, converters_map, enums_map);

        // Create token stream for chumsky
        let len = expression.len();
        let token_stream: Vec<(Token, std::ops::Range<usize>)> = tokens;

        // Parse
        let result = chumsky_parser.parse(chumsky::Stream::from_iter(
            len..len + 1,
            token_stream.into_iter(),
        ));

        match result {
            Ok(ast) => {
                parser.ast = Some(ast);
            }
            Err(errs) => {
                for err in errs {
                    parser.errors.push(format!("Parse error: {:?}", err));
                }
            }
        }

        parser
    }
}

/// Implementation of the [`OttlParser`] trait for [`Parser`].
impl OttlParser for Parser {
    fn is_error(&self) -> Result<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.join("; ").into())
        }
    }

    fn execute(&self, ctx: &mut EvalContext) -> Result<Value> {
        let ast = self
            .ast
            .as_ref()
            .ok_or_else(|| -> BoxError { "No AST available (parsing failed)".into() })?;

        evaluate_root(ast, ctx, &self.path_resolver)
    }
}

// =====================================================================================================================
// Modular Parser Components
// =====================================================================================================================

/// Type alias for parser error
type ParserError<'a> = Simple<Token<'a>>;

/// Parser for string literals
fn string_literal_parser<'a>(
) -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone {
    select! {
        Token::StringLiteral(s) => {
            let inner = &s[1..s.len()-1];
            let unescaped = inner.replace("\\\"", "\"").replace("\\\\", "\\");
            Value::String(unescaped)
        }
    }
}

/// Parser for integer literals
fn int_literal_parser<'a>() -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone
{
    select! {
        Token::IntLiteral(s) => {
            let val: i64 = s.parse().unwrap_or(0);
            Value::Int(val)
        }
    }
}

/// Parser for float literals
fn float_literal_parser<'a>(
) -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone {
    select! {
        Token::FloatLiteral(s) => {
            let val: f64 = s.parse().unwrap_or(0.0);
            Value::Float(val)
        }
    }
}

/// Parser for bytes literals
fn bytes_literal_parser<'a>(
) -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone {
    select! {
        Token::BytesLiteral(s) => {
            let hex = &s[2..];
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| {
                    let end = (i + 2).min(hex.len());
                    u8::from_str_radix(&hex[i..end], 16).unwrap_or(0)
                })
                .collect();
            Value::Bytes(bytes)
        }
    }
}

/// Parser for boolean literals
fn bool_literal_parser<'a>() -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone
{
    select! {
        Token::True => Value::Bool(true),
        Token::False => Value::Bool(false),
    }
}

/// Parser for nil literal
fn nil_literal_parser<'a>() -> impl ChumskyParser<Token<'a>, Value, Error = ParserError<'a>> + Clone
{
    just(Token::Nil).to(Value::Nil)
}

/// Combined parser for all literal types, returning ValueExpr
fn literal_parser<'a>() -> impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone
{
    choice((
        float_literal_parser(),
        int_literal_parser(),
        string_literal_parser(),
        bytes_literal_parser(),
        bool_literal_parser(),
        nil_literal_parser(),
    ))
    .map(ValueExpr::Literal)
}

/// Parser for lowercase identifier
fn lower_ident_parser<'a>() -> impl ChumskyParser<Token<'a>, String, Error = ParserError<'a>> + Clone
{
    select! {
        Token::LowerIdent(s) => s.to_string(),
    }
}

/// Parser for uppercase identifier
fn upper_ident_parser<'a>() -> impl ChumskyParser<Token<'a>, String, Error = ParserError<'a>> + Clone
{
    select! {
        Token::UpperIdent(s) => s.to_string(),
    }
}

/// Parser for any identifier (lower or upper)
fn ident_parser<'a>() -> impl ChumskyParser<Token<'a>, String, Error = ParserError<'a>> + Clone {
    lower_ident_parser().or(upper_ident_parser())
}

/// Parser for index expression: "[" (string | int) "]"
fn index_parser<'a>() -> impl ChumskyParser<Token<'a>, IndexExpr, Error = ParserError<'a>> + Clone {
    let string_index = select! {
        Token::StringLiteral(s) => {
            let inner = &s[1..s.len()-1];
            IndexExpr::String(inner.replace("\\\"", "\"").replace("\\\\", "\\"))
        }
    };

    let int_index = select! {
        Token::IntLiteral(s) => {
            let val: i64 = s.parse().unwrap_or(0);
            IndexExpr::Int(val)
        }
    };

    string_index
        .or(int_index)
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
}

/// Parser for path expression: lower_ident ("." ident)* index*
fn path_parser<'a>() -> impl ChumskyParser<Token<'a>, PathExpr, Error = ParserError<'a>> + Clone {
    lower_ident_parser()
        .then(just(Token::Dot).ignore_then(ident_parser()).repeated())
        .then(index_parser().repeated())
        .map(|((first, rest), indexes)| {
            let mut segments = vec![first];
            segments.extend(rest);
            PathExpr { segments, indexes }
        })
}

/// Parser for enum values (uppercase identifier resolved from enum map)
fn enum_parser<'a>(
    enums: std::collections::HashMap<String, i64>,
) -> impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone {
    upper_ident_parser().try_map(move |name, span| {
        if let Some(&val) = enums.get(&name) {
            Ok(ValueExpr::Literal(Value::Int(val)))
        } else {
            Err(Simple::custom(span, format!("Unknown enum: {}", name)))
        }
    })
}

/// Parser for comparison operators
fn comp_op_parser<'a>() -> impl ChumskyParser<Token<'a>, CompOp, Error = ParserError<'a>> + Clone {
    choice((
        just(Token::Eq).to(CompOp::Eq),
        just(Token::NotEq).to(CompOp::NotEq),
        just(Token::LessEq).to(CompOp::LessEq),
        just(Token::GreaterEq).to(CompOp::GreaterEq),
        just(Token::Less).to(CompOp::Less),
        just(Token::Greater).to(CompOp::Greater),
    ))
}

/// Parser for additive operators (+ -)
fn add_op_parser<'a>() -> impl ChumskyParser<Token<'a>, MathOp, Error = ParserError<'a>> + Clone {
    choice((
        just(Token::Plus).to(MathOp::Add),
        just(Token::Minus).to(MathOp::Sub),
    ))
}

/// Parser for multiplicative operators (* /)
fn mul_op_parser<'a>() -> impl ChumskyParser<Token<'a>, MathOp, Error = ParserError<'a>> + Clone {
    choice((
        just(Token::Star).to(MathOp::Mul),
        just(Token::Slash).to(MathOp::Div),
    ))
}

/// Parser for unary sign operators (returns true for +, false for -)
fn unary_sign_parser<'a>() -> impl ChumskyParser<Token<'a>, bool, Error = ParserError<'a>> + Clone {
    choice((just(Token::Plus).to(true), just(Token::Minus).to(false)))
}

/// Parser for numeric literal as MathExpr::Primary
fn math_numeric_literal_parser<'a>(
) -> impl ChumskyParser<Token<'a>, MathExpr, Error = ParserError<'a>> + Clone {
    choice((
        select! {
            Token::FloatLiteral(s) => {
                let val: f64 = s.parse().unwrap_or(0.0);
                MathExpr::Primary(ValueExpr::Literal(Value::Float(val)))
            }
        },
        select! {
            Token::IntLiteral(s) => {
                let val: i64 = s.parse().unwrap_or(0);
                MathExpr::Primary(ValueExpr::Literal(Value::Int(val)))
            }
        },
    ))
}

// =====================================================================================================================
// Chumsky Parser Builder - Main Entry Point
// =====================================================================================================================

/// Build the chumsky parser for OTTL
fn build_parser<'a>(
    editors_map: &'a mut CallbackMap,
    converters_map: &'a mut CallbackMap,
    enums_map: &'a mut EnumMap,
) -> impl ChumskyParser<Token<'a>, RootExpr, Error = ParserError<'a>> + 'a {
    // Clone maps for use in closures
    let enums_clone = enums_map.clone();
    let editors_clone = editors_map.clone();
    let converters_clone = converters_map.clone();

    // Create value expression parser (recursive)
    let value_expr = build_value_expr_parser(enums_clone.clone(), converters_clone.clone());

    // Create math expression parser for comparisons
    let math_expr_for_comparison = build_math_expr_parser(value_expr.clone());

    // Create boolean expression parser
    let bool_expr = build_bool_expr_parser(value_expr.clone(), math_expr_for_comparison);

    // Create math expression parser for root
    let math_expr = build_math_expr_parser(value_expr.clone());

    // Create editor statement parser
    let editor_statement =
        build_editor_statement_parser(value_expr, editors_clone, bool_expr.clone());

    // Root: editor_statement | boolean_expression | math_expression
    choice((
        editor_statement,
        bool_expr.map(RootExpr::BooleanExpression),
        math_expr.map(RootExpr::MathExpression),
    ))
    .then_ignore(end())
}

/// Build the value expression parser (handles recursion for lists, maps, and function calls)
fn build_value_expr_parser<'a>(
    enums: std::collections::HashMap<String, i64>,
    converters: std::collections::HashMap<String, CallbackFn>,
) -> impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone + 'a {
    recursive(move |value_expr| {
        // List: "[" (value ("," value)*)? "]"
        let list = value_expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(ValueExpr::List);

        // Map entry: string ":" value
        let map_entry = select! {
            Token::StringLiteral(s) => {
                let inner = &s[1..s.len()-1];
                inner.replace("\\\"", "\"").replace("\\\\", "\\")
            }
        }
        .then_ignore(just(Token::Colon))
        .then(value_expr.clone());

        // Map: "{" (map_entry ("," map_entry)*)? "}"
        let map = map_entry
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(ValueExpr::Map);

        // Arguments for function calls
        let named_arg = lower_ident_parser()
            .then_ignore(just(Token::Assign))
            .then(value_expr.clone())
            .map(|(name, value)| ArgExpr::Named { name, value });

        let positional_arg = value_expr.clone().map(ArgExpr::Positional);

        let arg = named_arg.or(positional_arg);
        let arg_list = arg.separated_by(just(Token::Comma)).allow_trailing();

        // Converter invocation: upper_ident "(" arg_list ")" index*
        let converter_call = {
            let converters = converters.clone();
            upper_ident_parser()
                .then(arg_list.delimited_by(just(Token::LParen), just(Token::RParen)))
                .then(index_parser().repeated())
                .map(move |((name, args), indexes)| {
                    let callback = converters.get(&name).cloned();
                    ValueExpr::FunctionCall(Box::new(FunctionCall {
                        name,
                        is_editor: false,
                        args,
                        indexes,
                        callback,
                    }))
                })
        };

        // Value expression: converter_call | list | map | enum | path | literal
        choice((
            converter_call,
            list,
            map,
            enum_parser(enums.clone()),
            path_parser().map(ValueExpr::Path),
            literal_parser(),
        ))
    })
}

/// Build the math expression parser with proper operator precedence
fn build_math_expr_parser<'a>(
    value_expr: impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone + 'a,
) -> impl ChumskyParser<Token<'a>, MathExpr, Error = ParserError<'a>> + Clone + 'a {
    recursive(move |math_expr| {
        // Parenthesized math expression
        let paren_math = math_expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Value that can be used in math (converter or path)
        let math_value = value_expr.clone().try_map(|v, span| {
            if let ValueExpr::FunctionCall(_) = &v {
                Ok(MathExpr::Primary(v))
            } else if let ValueExpr::Path(_) = &v {
                Ok(MathExpr::Primary(v))
            } else {
                Err(Simple::custom(span, "Expected converter or path"))
            }
        });

        // Primary: parenthesized | numeric literal | converter/path
        let primary = choice((paren_math, math_numeric_literal_parser(), math_value));

        // Factor with optional unary sign: [("+" | "-")] primary
        let factor = unary_sign_parser()
            .or_not()
            .then(primary)
            .map(|(op, expr)| match op {
                Some(false) => MathExpr::Negate(Box::new(expr)),
                _ => expr,
            });

        // Term: factor (("*" | "/") factor)*
        let term = factor
            .clone()
            .then(mul_op_parser().then(factor).repeated())
            .foldl(|left, (op, right)| MathExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });

        // Expression: term (("+" | "-") term)*
        term.clone()
            .then(add_op_parser().then(term).repeated())
            .foldl(|left, (op, right)| MathExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
    })
}

/// Build the boolean expression parser with proper operator precedence
fn build_bool_expr_parser<'a>(
    value_expr: impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone + 'a,
    math_expr: impl ChumskyParser<Token<'a>, MathExpr, Error = ParserError<'a>> + Clone + 'a,
) -> impl ChumskyParser<Token<'a>, BoolExpr, Error = ParserError<'a>> + Clone + 'a {
    // Convert math expression to value expression for comparisons
    let comparison_value = math_expr.map(|m| ValueExpr::Math(Box::new(m)));

    recursive(move |bool_expr| {
        // Boolean literals
        let bool_literal = select! {
            Token::True => BoolExpr::Literal(true),
            Token::False => BoolExpr::Literal(false),
        };

        // Comparison: value comp_op value
        let comparison = comparison_value
            .clone()
            .then(comp_op_parser())
            .then(comparison_value.clone())
            .map(|((left, op), right)| BoolExpr::Comparison { left, op, right });

        // Converter call in boolean context
        let bool_converter = value_expr.clone().try_map(|v, span| {
            if let ValueExpr::FunctionCall(fc) = v {
                Ok(BoolExpr::Converter(fc))
            } else {
                Err(Simple::custom(span, "Expected converter call"))
            }
        });

        // Path that evaluates to boolean
        let bool_path = path_parser().map(BoolExpr::Path);

        // Primary: parenthesized | comparison | literal | converter | path
        let bool_primary = choice((
            bool_expr
                .clone()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
            comparison,
            bool_literal,
            bool_converter,
            bool_path,
        ));

        // Factor with optional NOT: ["not"] primary
        let bool_factor = just(Token::Not)
            .or_not()
            .then(bool_primary)
            .map(|(not, expr)| {
                if not.is_some() {
                    BoolExpr::Not(Box::new(expr))
                } else {
                    expr
                }
            });

        // Term: factor ("and" factor)*
        let bool_term = bool_factor
            .clone()
            .then(just(Token::And).ignore_then(bool_factor).repeated())
            .foldl(|left, right| BoolExpr::And(Box::new(left), Box::new(right)));

        // Expression: term ("or" term)*
        bool_term
            .clone()
            .then(just(Token::Or).ignore_then(bool_term).repeated())
            .foldl(|left, right| BoolExpr::Or(Box::new(left), Box::new(right)))
    })
}

/// Build the editor statement parser
fn build_editor_statement_parser<'a>(
    value_expr: impl ChumskyParser<Token<'a>, ValueExpr, Error = ParserError<'a>> + Clone + 'a,
    editors: std::collections::HashMap<String, CallbackFn>,
    bool_expr: impl ChumskyParser<Token<'a>, BoolExpr, Error = ParserError<'a>> + Clone + 'a,
) -> impl ChumskyParser<Token<'a>, RootExpr, Error = ParserError<'a>> + 'a {
    // Arguments for editor calls
    let named_arg = lower_ident_parser()
        .then_ignore(just(Token::Assign))
        .then(value_expr.clone())
        .map(|(name, value)| ArgExpr::Named { name, value });

    let positional_arg = value_expr.map(ArgExpr::Positional);

    let arg = named_arg.or(positional_arg);
    let arg_list = arg.separated_by(just(Token::Comma)).allow_trailing();

    // Editor call: lower_ident "(" arg_list ")"
    let editor_call = lower_ident_parser()
        .then(arg_list.delimited_by(just(Token::LParen), just(Token::RParen)))
        .map(move |(name, args)| FunctionCall {
            callback: editors.get(&name).cloned(),
            name,
            is_editor: true,
            args,
            indexes: Vec::new(),
        });

    // WHERE clause: "where" bool_expr
    let where_clause = just(Token::Where).ignore_then(bool_expr);

    // Editor statement: editor_call [where_clause]
    editor_call
        .then(where_clause.or_not())
        .map(|(editor, condition)| RootExpr::EditorStatement(EditorStatement { editor, condition }))
}

// =====================================================================================================================
// AST Evaluation
// =====================================================================================================================

/// Evaluate the root expression
fn evaluate_root(root: &RootExpr, ctx: &mut EvalContext, resolver: &PathResolver) -> Result<Value> {
    match root {
        RootExpr::EditorStatement(stmt) => {
            // Check condition if present
            let should_execute = if let Some(ref cond) = stmt.condition {
                evaluate_bool_expr(cond, ctx, resolver)?
            } else {
                true
            };

            if should_execute {
                evaluate_function_call(&stmt.editor, ctx, resolver)?;
            }

            Ok(Value::Nil)
        }
        RootExpr::BooleanExpression(expr) => {
            let result = evaluate_bool_expr(expr, ctx, resolver)?;
            Ok(Value::Bool(result))
        }
        RootExpr::MathExpression(expr) => evaluate_math_expr(expr, ctx, resolver),
    }
}

/// Evaluate a boolean expression
fn evaluate_bool_expr(
    expr: &BoolExpr,
    ctx: &mut EvalContext,
    resolver: &PathResolver,
) -> Result<bool> {
    match expr {
        BoolExpr::Literal(b) => Ok(*b),
        BoolExpr::Comparison { left, op, right } => {
            let left_val = evaluate_value_expr(left, ctx, resolver)?;
            let right_val = evaluate_value_expr(right, ctx, resolver)?;
            evaluate_comparison(&left_val, op, &right_val)
        }
        BoolExpr::Converter(fc) => {
            let result = evaluate_function_call(fc, ctx, resolver)?;
            match result {
                Value::Bool(b) => Ok(b),
                _ => Err("Converter did not return a boolean".into()),
            }
        }
        BoolExpr::Path(path) => {
            let value = evaluate_path(path, ctx, resolver)?;
            match value {
                Value::Bool(b) => Ok(b),
                _ => Err("Path did not return a boolean".into()),
            }
        }
        BoolExpr::Not(inner) => {
            let result = evaluate_bool_expr(inner, ctx, resolver)?;
            Ok(!result)
        }
        BoolExpr::And(left, right) => {
            let left_result = evaluate_bool_expr(left, ctx, resolver)?;
            if !left_result {
                return Ok(false); // Short-circuit
            }
            evaluate_bool_expr(right, ctx, resolver)
        }
        BoolExpr::Or(left, right) => {
            let left_result = evaluate_bool_expr(left, ctx, resolver)?;
            if left_result {
                return Ok(true); // Short-circuit
            }
            evaluate_bool_expr(right, ctx, resolver)
        }
    }
}

/// Evaluate a comparison
fn evaluate_comparison(left: &Value, op: &CompOp, right: &Value) -> Result<bool> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        (Value::Float(l), Value::Float(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        (Value::Int(l), Value::Float(r)) => {
            let l = *l as f64;
            Ok(match op {
                CompOp::Eq => l == *r,
                CompOp::NotEq => l != *r,
                CompOp::Less => l < *r,
                CompOp::Greater => l > *r,
                CompOp::LessEq => l <= *r,
                CompOp::GreaterEq => l >= *r,
            })
        }
        (Value::Float(l), Value::Int(r)) => {
            let r = *r as f64;
            Ok(match op {
                CompOp::Eq => *l == r,
                CompOp::NotEq => *l != r,
                CompOp::Less => *l < r,
                CompOp::Greater => *l > r,
                CompOp::LessEq => *l <= r,
                CompOp::GreaterEq => *l >= r,
            })
        }
        (Value::String(l), Value::String(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        (Value::Bool(l), Value::Bool(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            _ => return Err("Boolean comparison only supports == and !=".into()),
        }),
        (Value::Nil, Value::Nil) => Ok(match op {
            CompOp::Eq => true,
            CompOp::NotEq => false,
            _ => return Err("Nil comparison only supports == and !=".into()),
        }),
        (Value::Nil, _) | (_, Value::Nil) => Ok(match op {
            CompOp::Eq => false,
            CompOp::NotEq => true,
            _ => return Err("Nil comparison only supports == and !=".into()),
        }),
        _ => Err(format!(
            "Cannot compare {:?} with {:?}",
            std::mem::discriminant(left),
            std::mem::discriminant(right)
        )
        .into()),
    }
}

/// Evaluate a value expression
fn evaluate_value_expr(
    expr: &ValueExpr,
    ctx: &mut EvalContext,
    resolver: &PathResolver,
) -> Result<Value> {
    match expr {
        ValueExpr::Literal(v) => Ok(v.clone()),
        ValueExpr::Path(path) => evaluate_path(path, ctx, resolver),
        ValueExpr::List(items) => {
            let values: Result<Vec<Value>> = items
                .iter()
                .map(|item| evaluate_value_expr(item, ctx, resolver))
                .collect();
            Ok(Value::List(values?))
        }
        ValueExpr::Map(entries) => {
            let mut map = std::collections::HashMap::new();
            for (key, value_expr) in entries {
                let value = evaluate_value_expr(value_expr, ctx, resolver)?;
                map.insert(key.clone(), value);
            }
            Ok(Value::Map(map))
        }
        ValueExpr::FunctionCall(fc) => evaluate_function_call(fc, ctx, resolver),
        ValueExpr::Math(math) => evaluate_math_expr(math, ctx, resolver),
    }
}

/// Evaluate a path expression
fn evaluate_path(path: &PathExpr, ctx: &mut EvalContext, resolver: &PathResolver) -> Result<Value> {
    // Build the path string
    let path_str = path.segments.join(".");

    // Get the accessor
    let accessor = resolver(&path_str)?;

    // Get the value
    let value = accessor.get(ctx, &path_str)?;

    // Apply indexes if any
    let mut current = value.clone();
    for index in &path.indexes {
        current = apply_index(&current, index)?;
    }

    Ok(current)
}

/// Apply an index to a value
fn apply_index(value: &Value, index: &IndexExpr) -> Result<Value> {
    match (value, index) {
        (Value::List(list), IndexExpr::Int(i)) => {
            let idx = if *i < 0 {
                (list.len() as i64 + i) as usize
            } else {
                *i as usize
            };
            list.get(idx)
                .cloned()
                .ok_or_else(|| format!("Index {} out of bounds", i).into())
        }
        (Value::Map(map), IndexExpr::String(key)) => map
            .get(key)
            .cloned()
            .ok_or_else(|| format!("Key '{}' not found", key).into()),
        (Value::String(s), IndexExpr::Int(i)) => {
            let idx = if *i < 0 {
                (s.len() as i64 + i) as usize
            } else {
                *i as usize
            };
            s.chars()
                .nth(idx)
                .map(|c| Value::String(c.to_string()))
                .ok_or_else(|| format!("Index {} out of bounds", i).into())
        }
        _ => Err(format!("Cannot index {:?} with {:?}", value, index).into()),
    }
}

/// Evaluate a function call
fn evaluate_function_call(
    fc: &FunctionCall,
    ctx: &mut EvalContext,
    resolver: &PathResolver,
) -> Result<Value> {
    // Evaluate arguments
    let mut args = Vec::new();
    for arg_expr in &fc.args {
        let arg = match arg_expr {
            ArgExpr::Positional(value_expr) => {
                let value = evaluate_value_expr(value_expr, ctx, resolver)?;
                Argument::Positional(value)
            }
            ArgExpr::Named { name, value } => {
                let evaluated = evaluate_value_expr(value, ctx, resolver)?;
                Argument::Named {
                    name: name.clone(),
                    value: evaluated,
                }
            }
        };
        args.push(arg);
    }

    // Call the callback
    let callback = fc
        .callback
        .as_ref()
        .ok_or_else(|| -> BoxError { format!("Unknown function: {}", fc.name).into() })?;

    let result = callback(ctx, args)?;

    // Apply indexes if any
    let mut current = result;
    for index in &fc.indexes {
        current = apply_index(&current, index)?;
    }

    Ok(current)
}

/// Evaluate a math expression
fn evaluate_math_expr(
    expr: &MathExpr,
    ctx: &mut EvalContext,
    resolver: &PathResolver,
) -> Result<Value> {
    match expr {
        MathExpr::Primary(value_expr) => evaluate_value_expr(value_expr, ctx, resolver),
        MathExpr::Negate(inner) => {
            let value = evaluate_math_expr(inner, ctx, resolver)?;
            match value {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err("Cannot negate non-numeric value".into()),
            }
        }
        MathExpr::Binary { left, op, right } => {
            let left_val = evaluate_math_expr(left, ctx, resolver)?;
            let right_val = evaluate_math_expr(right, ctx, resolver)?;
            evaluate_math_op(&left_val, op, &right_val)
        }
    }
}

/// Evaluate a math operation
fn evaluate_math_op(left: &Value, op: &MathOp, right: &Value) -> Result<Value> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => match op {
            MathOp::Add => Ok(Value::Int(l + r)),
            MathOp::Sub => Ok(Value::Int(l - r)),
            MathOp::Mul => Ok(Value::Int(l * r)),
            MathOp::Div => {
                if *r == 0 {
                    Err("Division by zero".into())
                } else {
                    Ok(Value::Int(l / r))
                }
            }
        },
        (Value::Float(l), Value::Float(r)) => match op {
            MathOp::Add => Ok(Value::Float(l + r)),
            MathOp::Sub => Ok(Value::Float(l - r)),
            MathOp::Mul => Ok(Value::Float(l * r)),
            MathOp::Div => {
                if *r == 0.0 {
                    Err("Division by zero".into())
                } else {
                    Ok(Value::Float(l / r))
                }
            }
        },
        (Value::Int(l), Value::Float(r)) => {
            let l = *l as f64;
            match op {
                MathOp::Add => Ok(Value::Float(l + r)),
                MathOp::Sub => Ok(Value::Float(l - r)),
                MathOp::Mul => Ok(Value::Float(l * r)),
                MathOp::Div => {
                    if *r == 0.0 {
                        Err("Division by zero".into())
                    } else {
                        Ok(Value::Float(l / r))
                    }
                }
            }
        }
        (Value::Float(l), Value::Int(r)) => {
            let r = *r as f64;
            match op {
                MathOp::Add => Ok(Value::Float(l + r)),
                MathOp::Sub => Ok(Value::Float(l - r)),
                MathOp::Mul => Ok(Value::Float(l * r)),
                MathOp::Div => {
                    if r == 0.0 {
                        Err("Division by zero".into())
                    } else {
                        Ok(Value::Float(l / r))
                    }
                }
            }
        }
        (Value::String(l), Value::String(r)) if matches!(op, MathOp::Add) => {
            Ok(Value::String(format!("{}{}", l, r)))
        }
        _ => Err(format!(
            "Cannot perform math operation on {:?} and {:?}",
            std::mem::discriminant(left),
            std::mem::discriminant(right)
        )
        .into()),
    }
}
