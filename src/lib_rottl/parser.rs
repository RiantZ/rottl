//! OTTL Parser Implementation
//!
//! This module contains the Parser struct and its implementation using chumsky 0.12.

use chumsky::prelude::*;
use chumsky::Parser as ChumskyParser;
use std::sync::Arc;

use super::lexer::Token;
use super::{
    Args, BoxError, CallbackFn, CallbackMap, EnumMap, EvalContext, OttlParser, PathAccessor,
    PathResolver, Result, Value,
};

// =====================================================================================================================
// Arena-based AST Types
// =====================================================================================================================

/// Index into the arena for BoolExpr nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolExprRef(u32);

/// Index into the arena for MathExpr nodes  
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MathExprRef(u32);

/// Index into the arena for ValueExpr nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueExprRef(u32);

/// Index into the arena for FunctionCall nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionCallRef(u32);

/// Arena storage for all AST nodes - enables cache-friendly traversal
#[derive(Debug, Clone, Default)]
pub struct AstArena {
    pub bool_exprs: Vec<ArenaBoolExpr>,
    pub math_exprs: Vec<ArenaMathExpr>,
    pub value_exprs: Vec<ArenaValueExpr>,
    pub function_calls: Vec<ArenaFunctionCall>,
}

impl AstArena {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn alloc_bool(&mut self, expr: ArenaBoolExpr) -> BoolExprRef {
        let idx = self.bool_exprs.len() as u32;
        self.bool_exprs.push(expr);
        BoolExprRef(idx)
    }

    #[inline]
    pub fn alloc_math(&mut self, expr: ArenaMathExpr) -> MathExprRef {
        let idx = self.math_exprs.len() as u32;
        self.math_exprs.push(expr);
        MathExprRef(idx)
    }

    #[inline]
    pub fn alloc_value(&mut self, expr: ArenaValueExpr) -> ValueExprRef {
        let idx = self.value_exprs.len() as u32;
        self.value_exprs.push(expr);
        ValueExprRef(idx)
    }

    #[inline]
    pub fn alloc_func(&mut self, call: ArenaFunctionCall) -> FunctionCallRef {
        let idx = self.function_calls.len() as u32;
        self.function_calls.push(call);
        FunctionCallRef(idx)
    }

    #[inline]
    pub fn get_bool(&self, r: BoolExprRef) -> &ArenaBoolExpr {
        &self.bool_exprs[r.0 as usize]
    }

    #[inline]
    pub fn get_math(&self, r: MathExprRef) -> &ArenaMathExpr {
        &self.math_exprs[r.0 as usize]
    }

    #[inline]
    pub fn get_value(&self, r: ValueExprRef) -> &ArenaValueExpr {
        &self.value_exprs[r.0 as usize]
    }

    #[inline]
    pub fn get_func(&self, r: FunctionCallRef) -> &ArenaFunctionCall {
        &self.function_calls[r.0 as usize]
    }
}

/// Arena-based BoolExpr using indices instead of Box
#[derive(Debug, Clone)]
pub enum ArenaBoolExpr {
    Literal(bool),
    Comparison {
        left: ValueExprRef,
        op: CompOp,
        right: ValueExprRef,
    },
    Converter(FunctionCallRef),
    /// Path with pre-resolved accessor
    Path(ResolvedPath),
    Not(BoolExprRef),
    And(BoolExprRef, BoolExprRef),
    Or(BoolExprRef, BoolExprRef),
}

/// Arena-based MathExpr using indices instead of Box
#[derive(Debug, Clone)]
pub enum ArenaMathExpr {
    Primary(ValueExprRef),
    Negate(MathExprRef),
    Binary {
        left: MathExprRef,
        op: MathOp,
        right: MathExprRef,
    },
}

/// Resolved path with pre-computed path string and accessor (resolved at parse time)
#[derive(Clone)]
pub struct ResolvedPath {
    /// Pre-computed full path string (e.g., "my.int.value")
    pub full_path: String,
    /// Pre-resolved accessor (resolved once at parse time, not at each execution)
    pub accessor: Arc<dyn PathAccessor + Send + Sync>,
    /// Optional indexes for indexing into the result
    pub indexes: Vec<IndexExpr>,
}

impl std::fmt::Debug for ResolvedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedPath")
            .field("full_path", &self.full_path)
            .field("indexes", &self.indexes)
            .finish()
    }
}

/// Arena-based ValueExpr using indices instead of Box
#[derive(Debug, Clone)]
pub enum ArenaValueExpr {
    Literal(Value),
    /// Path with pre-resolved accessor (no runtime lookup!)
    Path(ResolvedPath),
    List(Vec<ValueExprRef>),
    Map(Vec<(String, ValueExprRef)>),
    FunctionCall(FunctionCallRef),
    Math(MathExprRef),
}

/// Arena-based FunctionCall using indices for args
#[derive(Clone)]
pub struct ArenaFunctionCall {
    pub name: String,
    pub is_editor: bool,
    pub args: Vec<ArenaArgExpr>,
    pub indexes: Vec<IndexExpr>,
    pub callback: Option<CallbackFn>,
}

impl std::fmt::Debug for ArenaFunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArenaFunctionCall")
            .field("name", &self.name)
            .field("is_editor", &self.is_editor)
            .field("args", &self.args)
            .field("indexes", &self.indexes)
            .field("callback", &self.callback.is_some())
            .finish()
    }
}

/// Arena-based ArgExpr
#[derive(Debug, Clone)]
pub enum ArenaArgExpr {
    Positional(ValueExprRef),
    Named { name: String, value: ValueExprRef },
}

/// Arena-based EditorStatement
#[derive(Debug, Clone)]
pub struct ArenaEditorStatement {
    pub editor: FunctionCallRef,
    pub condition: Option<BoolExprRef>,
}

/// Arena-based root expression
#[derive(Debug, Clone)]
pub enum ArenaRootExpr {
    EditorStatement(ArenaEditorStatement),
    BooleanExpression(BoolExprRef),
    MathExpression(MathExprRef),
}

// =====================================================================================================================
// Original AST Node Types (used during parsing, then converted to arena)
// =====================================================================================================================

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
// AST to Arena Conversion
// =====================================================================================================================

/// Helper to resolve a PathExpr to ResolvedPath at parse time
fn resolve_path(path: &PathExpr, resolver: &PathResolver) -> Result<ResolvedPath> {
    let full_path = path.segments.join(".");
    let accessor = resolver(&full_path)?;
    Ok(ResolvedPath {
        full_path,
        accessor,
        indexes: path.indexes.clone(),
    })
}

/// Convert boxed AST to arena-based AST for cache-friendly execution
/// Path resolution happens HERE (once at parse time), not at each execution!
fn convert_to_arena(
    root: &RootExpr,
    arena: &mut AstArena,
    resolver: &PathResolver,
) -> Result<ArenaRootExpr> {
    match root {
        RootExpr::EditorStatement(stmt) => {
            let editor_ref = convert_function_call(&stmt.editor, arena, resolver)?;
            let condition_ref = match &stmt.condition {
                Some(c) => Some(convert_bool_expr(c, arena, resolver)?),
                None => None,
            };
            Ok(ArenaRootExpr::EditorStatement(ArenaEditorStatement {
                editor: editor_ref,
                condition: condition_ref,
            }))
        }
        RootExpr::BooleanExpression(expr) => {
            let expr_ref = convert_bool_expr(expr, arena, resolver)?;
            Ok(ArenaRootExpr::BooleanExpression(expr_ref))
        }
        RootExpr::MathExpression(expr) => {
            let expr_ref = convert_math_expr(expr, arena, resolver)?;
            Ok(ArenaRootExpr::MathExpression(expr_ref))
        }
    }
}

/// Check if a ValueExpr is a literal and get its value
fn try_get_literal(expr: &ValueExpr) -> Option<&Value> {
    match expr {
        ValueExpr::Literal(v) => Some(v),
        _ => None,
    }
}

/// Check if arena bool expr is a literal
fn arena_bool_is_literal(arena: &AstArena, r: BoolExprRef) -> Option<bool> {
    match arena.get_bool(r) {
        ArenaBoolExpr::Literal(b) => Some(*b),
        _ => None,
    }
}

/// Evaluate comparison at compile time for two literal values
fn eval_comparison_const(left: &Value, op: &CompOp, right: &Value) -> Option<bool> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => Some(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        (Value::Float(l), Value::Float(r)) => Some(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        (Value::Int(l), Value::Float(r)) => {
            let l = *l as f64;
            Some(match op {
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
            Some(match op {
                CompOp::Eq => *l == r,
                CompOp::NotEq => *l != r,
                CompOp::Less => *l < r,
                CompOp::Greater => *l > r,
                CompOp::LessEq => *l <= r,
                CompOp::GreaterEq => *l >= r,
            })
        }
        (Value::Bool(l), Value::Bool(r)) => Some(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            _ => return None,
        }),
        (Value::String(l), Value::String(r)) => Some(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            CompOp::Less => l < r,
            CompOp::Greater => l > r,
            CompOp::LessEq => l <= r,
            CompOp::GreaterEq => l >= r,
        }),
        _ => None,
    }
}

fn convert_bool_expr(
    expr: &BoolExpr,
    arena: &mut AstArena,
    resolver: &PathResolver,
) -> Result<BoolExprRef> {
    let arena_expr = match expr {
        BoolExpr::Literal(b) => ArenaBoolExpr::Literal(*b),
        BoolExpr::Comparison { left, op, right } => {
            // CONSTANT FOLDING: if both sides are literals, compute at parse time!
            if let (Some(left_val), Some(right_val)) =
                (try_get_literal(left), try_get_literal(right))
            {
                if let Some(result) = eval_comparison_const(left_val, op, right_val) {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(result)));
                }
            }
            let left_ref = convert_value_expr(left, arena, resolver)?;
            let right_ref = convert_value_expr(right, arena, resolver)?;
            ArenaBoolExpr::Comparison {
                left: left_ref,
                op: *op,
                right: right_ref,
            }
        }
        BoolExpr::Converter(fc) => {
            let fc_ref = convert_function_call(fc, arena, resolver)?;
            ArenaBoolExpr::Converter(fc_ref)
        }
        BoolExpr::Path(path) => {
            // Resolve path at parse time!
            let resolved = resolve_path(path, resolver)?;
            ArenaBoolExpr::Path(resolved)
        }
        BoolExpr::Not(inner) => {
            let inner_ref = convert_bool_expr(inner, arena, resolver)?;
            // CONSTANT FOLDING: not(literal) => !literal
            if let Some(b) = arena_bool_is_literal(arena, inner_ref) {
                return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(!b)));
            }
            ArenaBoolExpr::Not(inner_ref)
        }
        BoolExpr::And(left, right) => {
            let left_ref = convert_bool_expr(left, arena, resolver)?;
            // CONSTANT FOLDING: false && x => false, true && x => x
            if let Some(left_val) = arena_bool_is_literal(arena, left_ref) {
                if !left_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(false)));
                }
                // left is true, result is just right
                return convert_bool_expr(right, arena, resolver);
            }
            let right_ref = convert_bool_expr(right, arena, resolver)?;
            // CONSTANT FOLDING: x && false => false, x && true => x
            if let Some(right_val) = arena_bool_is_literal(arena, right_ref) {
                if !right_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(false)));
                }
                // right is true, result is just left
                return Ok(left_ref);
            }
            ArenaBoolExpr::And(left_ref, right_ref)
        }
        BoolExpr::Or(left, right) => {
            let left_ref = convert_bool_expr(left, arena, resolver)?;
            // CONSTANT FOLDING: true || x => true, false || x => x
            if let Some(left_val) = arena_bool_is_literal(arena, left_ref) {
                if left_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(true)));
                }
                // left is false, result is just right
                return convert_bool_expr(right, arena, resolver);
            }
            let right_ref = convert_bool_expr(right, arena, resolver)?;
            // CONSTANT FOLDING: x || true => true, x || false => x
            if let Some(right_val) = arena_bool_is_literal(arena, right_ref) {
                if right_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(true)));
                }
                // right is false, result is just left
                return Ok(left_ref);
            }
            ArenaBoolExpr::Or(left_ref, right_ref)
        }
    };
    Ok(arena.alloc_bool(arena_expr))
}

/// Try to get literal value from a MathExpr
fn try_get_math_literal(expr: &MathExpr) -> Option<&Value> {
    match expr {
        MathExpr::Primary(ValueExpr::Literal(v)) => Some(v),
        _ => None,
    }
}

/// Evaluate math operation at compile time
fn eval_math_const(left: &Value, op: &MathOp, right: &Value) -> Option<Value> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => Some(match op {
            MathOp::Add => Value::Int(l + r),
            MathOp::Sub => Value::Int(l - r),
            MathOp::Mul => Value::Int(l * r),
            MathOp::Div => {
                if *r == 0 {
                    return None;
                }
                Value::Int(l / r)
            }
        }),
        (Value::Float(l), Value::Float(r)) => Some(match op {
            MathOp::Add => Value::Float(l + r),
            MathOp::Sub => Value::Float(l - r),
            MathOp::Mul => Value::Float(l * r),
            MathOp::Div => {
                if *r == 0.0 {
                    return None; // Don't fold division by zero - let runtime handle it
                }
                Value::Float(l / r)
            }
        }),
        (Value::Int(l), Value::Float(r)) => {
            let l = *l as f64;
            Some(match op {
                MathOp::Add => Value::Float(l + r),
                MathOp::Sub => Value::Float(l - r),
                MathOp::Mul => Value::Float(l * r),
                MathOp::Div => {
                    if *r == 0.0 {
                        return None;
                    }
                    Value::Float(l / r)
                }
            })
        }
        (Value::Float(l), Value::Int(r)) => {
            let r = *r as f64;
            Some(match op {
                MathOp::Add => Value::Float(l + r),
                MathOp::Sub => Value::Float(l - r),
                MathOp::Mul => Value::Float(l * r),
                MathOp::Div => {
                    if r == 0.0 {
                        return None;
                    }
                    Value::Float(l / r)
                }
            })
        }
        _ => None,
    }
}

fn convert_math_expr(
    expr: &MathExpr,
    arena: &mut AstArena,
    resolver: &PathResolver,
) -> Result<MathExprRef> {
    let arena_expr = match expr {
        MathExpr::Primary(v) => {
            let v_ref = convert_value_expr(v, arena, resolver)?;
            ArenaMathExpr::Primary(v_ref)
        }
        MathExpr::Negate(inner) => {
            // CONSTANT FOLDING: -literal => literal negated
            if let Some(val) = try_get_math_literal(inner) {
                let negated = match val {
                    Value::Int(i) => Some(Value::Int(-i)),
                    Value::Float(f) => Some(Value::Float(-f)),
                    _ => None,
                };
                if let Some(v) = negated {
                    let v_ref = arena.alloc_value(ArenaValueExpr::Literal(v));
                    return Ok(arena.alloc_math(ArenaMathExpr::Primary(v_ref)));
                }
            }
            let inner_ref = convert_math_expr(inner, arena, resolver)?;
            ArenaMathExpr::Negate(inner_ref)
        }
        MathExpr::Binary { left, op, right } => {
            // CONSTANT FOLDING: literal op literal => computed literal
            if let (Some(left_val), Some(right_val)) =
                (try_get_math_literal(left), try_get_math_literal(right))
            {
                if let Some(result) = eval_math_const(left_val, op, right_val) {
                    let v_ref = arena.alloc_value(ArenaValueExpr::Literal(result));
                    return Ok(arena.alloc_math(ArenaMathExpr::Primary(v_ref)));
                }
            }
            let left_ref = convert_math_expr(left, arena, resolver)?;
            let right_ref = convert_math_expr(right, arena, resolver)?;
            ArenaMathExpr::Binary {
                left: left_ref,
                op: *op,
                right: right_ref,
            }
        }
    };
    Ok(arena.alloc_math(arena_expr))
}

fn convert_value_expr(
    expr: &ValueExpr,
    arena: &mut AstArena,
    resolver: &PathResolver,
) -> Result<ValueExprRef> {
    let arena_expr = match expr {
        ValueExpr::Literal(v) => ArenaValueExpr::Literal(v.clone()),
        ValueExpr::Path(path) => {
            // Resolve path at parse time!
            let resolved = resolve_path(path, resolver)?;
            ArenaValueExpr::Path(resolved)
        }
        ValueExpr::List(items) => {
            let refs: Result<Vec<_>> = items
                .iter()
                .map(|i| convert_value_expr(i, arena, resolver))
                .collect();
            ArenaValueExpr::List(refs?)
        }
        ValueExpr::Map(entries) => {
            let refs: Result<Vec<_>> = entries
                .iter()
                .map(|(k, v)| Ok((k.clone(), convert_value_expr(v, arena, resolver)?)))
                .collect();
            ArenaValueExpr::Map(refs?)
        }
        ValueExpr::FunctionCall(fc) => {
            let fc_ref = convert_function_call(fc, arena, resolver)?;
            ArenaValueExpr::FunctionCall(fc_ref)
        }
        ValueExpr::Math(m) => {
            let m_ref = convert_math_expr(m, arena, resolver)?;
            ArenaValueExpr::Math(m_ref)
        }
    };
    Ok(arena.alloc_value(arena_expr))
}

fn convert_function_call(
    fc: &FunctionCall,
    arena: &mut AstArena,
    resolver: &PathResolver,
) -> Result<FunctionCallRef> {
    let args: Result<Vec<_>> = fc
        .args
        .iter()
        .map(|arg| match arg {
            ArgExpr::Positional(v) => Ok(ArenaArgExpr::Positional(convert_value_expr(
                v, arena, resolver,
            )?)),
            ArgExpr::Named { name, value } => Ok(ArenaArgExpr::Named {
                name: name.clone(),
                value: convert_value_expr(value, arena, resolver)?,
            }),
        })
        .collect();

    let arena_fc = ArenaFunctionCall {
        name: fc.name.clone(),
        is_editor: fc.is_editor,
        args: args?,
        indexes: fc.indexes.clone(),
        callback: fc.callback.clone(),
    };
    Ok(arena.alloc_func(arena_fc))
}

// =====================================================================================================================
// Parser Implementation
// =====================================================================================================================

/// OTTL Parser that parses input strings and produces executable objects.
/// Paths are resolved at parse time (not at execution time) for maximum performance.
pub struct Parser {
    /// Arena-based AST for cache-friendly execution
    arena: AstArena,
    /// Arena-based root expression
    arena_root: Option<ArenaRootExpr>,
    /// Parsing errors
    errors: Vec<String>,
}

impl Parser {
    /// Creates a new parser with the given configuration.
    pub fn new(
        editors_map: &CallbackMap,
        converters_map: &CallbackMap,
        enums_map: &EnumMap,
        path_resolver_cb: &PathResolver,
        expression: &str,
    ) -> Self {
        let mut parser = Parser {
            arena: AstArena::new(),
            arena_root: None,
            errors: Vec::new(),
        };

        // Tokenize the input
        let tokens_with_spans = match super::lexer::Lexer::collect_with_spans(expression) {
            Ok(tokens) => tokens,
            Err(e) => {
                parser.errors.push(format!("Lexer error: {}", e));
                return parser;
            }
        };

        if tokens_with_spans.is_empty() && !expression.trim().is_empty() {
            parser.errors.push("Lexer failed to tokenize input".into());
            return parser;
        }

        // Extract just the tokens for parsing
        let tokens: Vec<Token> = tokens_with_spans.into_iter().map(|(t, _)| t).collect();

        // Build the chumsky parser
        let chumsky_parser = build_parser(editors_map, converters_map, enums_map);

        // Parse using slice input
        let result = chumsky_parser.parse(&tokens[..]);

        match result.into_result() {
            Ok(ast) => {
                // Convert to arena-based AST for cache-friendly execution
                // Path resolution happens HERE (once), not at each execution!
                match convert_to_arena(&ast, &mut parser.arena, path_resolver_cb) {
                    Ok(arena_root) => {
                        parser.arena_root = Some(arena_root);
                    }
                    Err(e) => {
                        parser.errors.push(format!("Path resolution error: {}", e));
                    }
                }
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
        let arena_root = self
            .arena_root
            .as_ref()
            .ok_or_else(|| -> BoxError { "No AST available (parsing failed)".into() })?;

        // No runtime path resolution - paths are pre-resolved at parse time!
        arena_evaluate_root(arena_root, &self.arena, ctx)
    }
}

// =====================================================================================================================
// Chumsky 0.12 Parser Builder
// =====================================================================================================================

/// Type alias for our input - a slice of tokens
type TokenInput<'src> = &'src [Token<'src>];

/// Type alias for parser extra (default error handling)
type ParserExtra<'src> = extra::Err<Rich<'src, Token<'src>>>;

// =====================================================================================================================
// Parser Components (extracted for maintainability)
// =====================================================================================================================

/// Parser for literal values (string, int, float, bytes, bool, nil)
/// Supports optional unary minus for numeric literals
fn literal_parser<'a>(
) -> impl chumsky::Parser<'a, TokenInput<'a>, ValueExpr, ParserExtra<'a>> + Clone {
    let string_literal = select_ref! {
        Token::StringLiteral(s) => {
            let inner = &s[1..s.len()-1];
            let unescaped = inner.replace("\\\"", "\"").replace("\\\\", "\\");
            Value::string(unescaped)
        }
    };

    // Signed int literal: optional minus followed by int
    let int_literal = just(&Token::Minus)
        .or_not()
        .then(select_ref! {
            Token::IntLiteral(s) => s.parse::<i64>().unwrap_or(0)
        })
        .map(|(neg, val)| {
            let val = if neg.is_some() { -val } else { val };
            Value::Int(val)
        });

    // Signed float literal: optional minus followed by float
    let float_literal = just(&Token::Minus)
        .or_not()
        .then(select_ref! {
            Token::FloatLiteral(s) => s.parse::<f64>().unwrap_or(0.0)
        })
        .map(|(neg, val)| {
            let val = if neg.is_some() { -val } else { val };
            Value::Float(val)
        });

    let bytes_literal = select_ref! {
        Token::BytesLiteral(s) => {
            let hex = &s[2..];
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| {
                    let end = (i + 2).min(hex.len());
                    u8::from_str_radix(&hex[i..end], 16).unwrap_or(0)
                })
                .collect();
            Value::bytes(bytes)
        }
    };

    let bool_literal = select_ref! {
        Token::True => Value::Bool(true),
        Token::False => Value::Bool(false),
    };

    let nil_literal = just(&Token::Nil).to(Value::Nil);

    choice((
        float_literal,
        int_literal,
        string_literal,
        bytes_literal,
        bool_literal,
        nil_literal,
    ))
    .map(ValueExpr::Literal)
}

// =====================================================================================================================
/// Parser for index expressions: "[" (string | int) "]"
fn index_parser<'a>() -> impl chumsky::Parser<'a, TokenInput<'a>, IndexExpr, ParserExtra<'a>> + Clone
{
    let index_string = select_ref! {
        Token::StringLiteral(s) => {
            let inner = &s[1..s.len()-1];
            IndexExpr::String(inner.replace("\\\"", "\"").replace("\\\\", "\\"))
        }
    };

    let index_int = select_ref! {
        Token::IntLiteral(s) => {
            let val: i64 = s.parse().unwrap_or(0);
            IndexExpr::Int(val)
        }
    };

    index_string
        .or(index_int)
        .delimited_by(just(&Token::LBracket), just(&Token::RBracket))
}

// =====================================================================================================================
/// Parser for lowercase identifiers
fn lower_ident_parser<'a>(
) -> impl chumsky::Parser<'a, TokenInput<'a>, String, ParserExtra<'a>> + Clone {
    select_ref! {
        Token::LowerIdent(s) => s.to_string(),
    }
}

// =====================================================================================================================
/// Parser for uppercase identifiers
fn upper_ident_parser<'a>(
) -> impl chumsky::Parser<'a, TokenInput<'a>, String, ParserExtra<'a>> + Clone {
    select_ref! {
        Token::UpperIdent(s) => s.to_string(),
    }
}

// =====================================================================================================================
/// Parser for path expressions: lower_ident ("." ident_segment)* index*
fn path_parser<'a>() -> impl chumsky::Parser<'a, TokenInput<'a>, PathExpr, ParserExtra<'a>> + Clone
{
    let lower_ident = lower_ident_parser();
    let ident_segment = lower_ident_parser().or(upper_ident_parser());
    let index = index_parser();

    lower_ident
        .then(
            just(&Token::Dot)
                .ignore_then(ident_segment)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then(index.repeated().collect::<Vec<_>>())
        .map(|((first, rest), indexes)| {
            let mut segments = vec![first];
            segments.extend(rest);
            PathExpr { segments, indexes }
        })
}

// =====================================================================================================================
/// Parser for comparison operators
fn comp_op_parser<'a>() -> impl chumsky::Parser<'a, TokenInput<'a>, CompOp, ParserExtra<'a>> + Clone
{
    choice((
        just(&Token::Eq).to(CompOp::Eq),
        just(&Token::NotEq).to(CompOp::NotEq),
        just(&Token::LessEq).to(CompOp::LessEq),
        just(&Token::GreaterEq).to(CompOp::GreaterEq),
        just(&Token::Less).to(CompOp::Less),
        just(&Token::Greater).to(CompOp::Greater),
    ))
}

// =====================================================================================================================
/// Parser for argument list (used by both value_expr and editor_call)
/// Takes a value_expr parser that can handle math expressions
fn arg_list_parser<'a>(
    arg_value: impl chumsky::Parser<'a, TokenInput<'a>, ValueExpr, ParserExtra<'a>> + Clone + 'a,
) -> impl chumsky::Parser<'a, TokenInput<'a>, Vec<ArgExpr>, ParserExtra<'a>> + Clone + 'a {
    let lower_ident = lower_ident_parser();

    let named_arg = lower_ident
        .then_ignore(just(&Token::Assign))
        .then(arg_value.clone())
        .map(|(name, value)| ArgExpr::Named { name, value });

    let positional_arg = arg_value.map(ArgExpr::Positional);

    named_arg
        .or(positional_arg)
        .separated_by(just(&Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
}

// =====================================================================================================================
/// Wraps a MathExpr into ValueExpr, unwrapping simple Primary values
fn math_to_value_expr(math: MathExpr) -> ValueExpr {
    match math {
        // If it's just a primary (no operators), unwrap to the original ValueExpr
        MathExpr::Primary(v) => v,
        // Otherwise wrap as Math
        other => ValueExpr::Math(Box::new(other)),
    }
}

// =====================================================================================================================
/// Creates a math expression parser (shared logic for comparison and root contexts)
fn make_math_expr<'a>(
    value_expr: impl chumsky::Parser<'a, TokenInput<'a>, ValueExpr, ParserExtra<'a>> + Clone + 'a,
) -> impl chumsky::Parser<'a, TokenInput<'a>, MathExpr, ParserExtra<'a>> + Clone + 'a {
    recursive(move |math_expr| {
        let math_literal = choice((
            select_ref! {
                Token::FloatLiteral(s) => {
                    let val: f64 = s.parse().unwrap_or(0.0);
                    MathExpr::Primary(ValueExpr::Literal(Value::Float(val)))
                }
            },
            select_ref! {
                Token::IntLiteral(s) => {
                    let val: i64 = s.parse().unwrap_or(0);
                    MathExpr::Primary(ValueExpr::Literal(Value::Int(val)))
                }
            },
        ));

        let paren_math = math_expr
            .clone()
            .delimited_by(just(&Token::LParen), just(&Token::RParen));

        let math_value = value_expr.clone().try_map(|v, span| {
            match &v {
                ValueExpr::FunctionCall(_) => Ok(MathExpr::Primary(v)),
                ValueExpr::Path(_) => Ok(MathExpr::Primary(v)),
                ValueExpr::Literal(_) => Ok(MathExpr::Primary(v)),
                // Allow lists and maps for comparison operations (==, !=)
                ValueExpr::List(_) => Ok(MathExpr::Primary(v)),
                ValueExpr::Map(_) => Ok(MathExpr::Primary(v)),
                _ => Err(Rich::custom(
                    span,
                    "Expected converter, path, literal, list, or map",
                )),
            }
        });

        let primary = choice((paren_math, math_literal, math_value));

        let unary_op = choice((just(&Token::Plus).to(true), just(&Token::Minus).to(false)));

        let factor = unary_op.or_not().then(primary).map(|(op, expr)| match op {
            Some(false) => MathExpr::Negate(Box::new(expr)),
            _ => expr,
        });

        let mul_op = choice((
            just(&Token::Multiply).to(MathOp::Mul),
            just(&Token::Divide).to(MathOp::Div),
        ));

        let term = factor
            .clone()
            .foldl(mul_op.then(factor).repeated(), |left, (op, right)| {
                MathExpr::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                }
            });

        let add_op = choice((
            just(&Token::Plus).to(MathOp::Add),
            just(&Token::Minus).to(MathOp::Sub),
        ));

        term.clone()
            .foldl(add_op.then(term).repeated(), |left, (op, right)| {
                MathExpr::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                }
            })
    })
}

// =====================================================================================================================
// Main Parser Builder
// =====================================================================================================================
/// Build the chumsky parser for OTTL (chumsky 0.12 API)
fn build_parser<'a>(
    editors_map: &'a CallbackMap,
    converters_map: &'a CallbackMap,
    enums_map: &'a EnumMap,
) -> impl chumsky::Parser<'a, TokenInput<'a>, RootExpr, ParserExtra<'a>> + 'a {
    // References are Copy, so they can be captured in move closures without cloning

    // Use extracted parsers
    let literal = literal_parser();
    let index = index_parser();
    let path = path_parser();
    let comp_op = comp_op_parser();
    let lower_ident = lower_ident_parser();
    let upper_ident = upper_ident_parser();

    // Enum: uppercase identifier that's in the enum map
    let enum_parser = {
        upper_ident.clone().try_map(move |name, span| {
            if let Some(&val) = enums_map.get(&name) {
                Ok(ValueExpr::Literal(Value::Int(val)))
            } else {
                Err(Rich::custom(span, format!("Unknown enum: {}", name)))
            }
        })
    };

    // Clone path for bool_expr
    let path_for_bool = path.clone();

    // Value expression (recursive)
    let value_expr = recursive(|value_expr| {
        // List: "[" (value ("," value)*)? "]"
        let list = value_expr
            .clone()
            .separated_by(just(&Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(&Token::LBracket), just(&Token::RBracket))
            .map(ValueExpr::List);

        // Map entry: string ":" value
        let map_entry = select_ref! {
            Token::StringLiteral(s) => {
                let inner = &s[1..s.len()-1];
                inner.replace("\\\"", "\"").replace("\\\\", "\\")
            }
        }
        .then_ignore(just(&Token::Colon))
        .then(value_expr.clone());

        // Map: "{" (map_entry ("," map_entry)*)? "}"
        let map = map_entry
            .separated_by(just(&Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(&Token::LBrace), just(&Token::RBrace))
            .map(ValueExpr::Map);

        // Create arg_value parser that supports math expressions in arguments
        let arg_value = make_math_expr(value_expr.clone()).map(math_to_value_expr);

        // Argument list using math-aware parser
        let arg_list = arg_list_parser(arg_value);

        // Converter invocation: upper_ident "(" arg_list ")" index*
        let converter_call = {
            upper_ident
                .clone()
                .then(arg_list.delimited_by(just(&Token::LParen), just(&Token::RParen)))
                .then(index.clone().repeated().collect::<Vec<_>>())
                .map(move |((name, args), indexes)| {
                    let callback = converters_map.get(&name).cloned();
                    ValueExpr::FunctionCall(Box::new(FunctionCall {
                        name,
                        is_editor: false,
                        args,
                        indexes,
                        callback,
                    }))
                })
        };

        choice((
            converter_call,
            list,
            map,
            enum_parser.clone(),
            path.clone().map(ValueExpr::Path),
            literal.clone(),
        ))
    });

    // Editor call using math-aware arg_list_parser
    let editor_call = {
        // Create arg_value parser that supports math expressions in editor arguments
        let arg_value = make_math_expr(value_expr.clone()).map(math_to_value_expr);
        let arg_list = arg_list_parser(arg_value);

        lower_ident
            .clone()
            .then(arg_list.delimited_by(just(&Token::LParen), just(&Token::RParen)))
            .map(move |(name, args)| FunctionCall {
                callback: editors_map.get(&name).cloned(),
                name,
                is_editor: true,
                args,
                indexes: Vec::new(),
            })
    };

    // Math expression using extracted make_math_expr (eliminates duplication)
    let math_expr_for_comparison = make_math_expr(value_expr.clone());
    let comparison_value = math_expr_for_comparison
        .clone()
        .map(|m| ValueExpr::Math(Box::new(m)));

    // Boolean expression
    let bool_expr = recursive({
        let value_expr = value_expr.clone();
        let comparison_value = comparison_value.clone();
        let path = path_for_bool.clone();
        move |bool_expr| {
            let bool_literal_expr = select_ref! {
                Token::True => BoolExpr::Literal(true),
                Token::False => BoolExpr::Literal(false),
            };

            let comparison = comparison_value
                .clone()
                .then(comp_op.clone())
                .then(comparison_value.clone())
                .map(|((left, op), right)| BoolExpr::Comparison { left, op, right });

            let bool_converter = value_expr.clone().try_map(|v, span| {
                if let ValueExpr::FunctionCall(fc) = v {
                    Ok(BoolExpr::Converter(fc))
                } else {
                    Err(Rich::custom(span, "Expected converter call"))
                }
            });

            let bool_path = path.clone().map(BoolExpr::Path);

            let bool_primary = choice((
                bool_expr
                    .clone()
                    .delimited_by(just(&Token::LParen), just(&Token::RParen)),
                comparison,
                bool_literal_expr,
                bool_converter,
                bool_path,
            ));

            let bool_factor = just(&Token::Not)
                .or_not()
                .then(bool_primary)
                .map(|(not, expr)| {
                    if not.is_some() {
                        BoolExpr::Not(Box::new(expr))
                    } else {
                        expr
                    }
                });

            let bool_term = bool_factor.clone().foldl(
                just(&Token::And).ignore_then(bool_factor).repeated(),
                |left, right| BoolExpr::And(Box::new(left), Box::new(right)),
            );

            bool_term.clone().foldl(
                just(&Token::Or).ignore_then(bool_term).repeated(),
                |left, right| BoolExpr::Or(Box::new(left), Box::new(right)),
            )
        }
    });

    // Math expression for root (reuses make_math_expr)
    let math_expr = make_math_expr(value_expr.clone());

    // Math expression that requires at least one binary operation
    // This ensures we try math_expr before bool_expr for expressions like "Sum(1,2) + 3"
    let math_expr_with_ops = math_expr.clone().try_map(|m, span| {
        // Only accept if it's a binary expression (has operators)
        if let MathExpr::Binary { .. } = &m {
            Ok(m)
        } else if let MathExpr::Negate(_) = &m {
            // Also accept unary negation
            Ok(m)
        } else {
            Err(Rich::custom(
                span,
                "Expected math expression with operators",
            ))
        }
    });

    // WHERE clause
    let where_clause = just(&Token::Where).ignore_then(bool_expr.clone());

    // Editor statement
    let editor_statement = editor_call
        .then(where_clause.or_not())
        .map(|(editor, condition)| {
            RootExpr::EditorStatement(EditorStatement { editor, condition })
        });

    // Root expression
    // Each alternative includes end() to ensure full input is consumed.
    // This enables backtracking: if math_expr_with_ops parses "a + b" but
    // "== c" remains, end() fails and bool_expr is tried instead.
    choice((
        editor_statement.then_ignore(end()),
        math_expr_with_ops
            .map(RootExpr::MathExpression)
            .then_ignore(end()),
        bool_expr
            .map(RootExpr::BooleanExpression)
            .then_ignore(end()),
        math_expr.map(RootExpr::MathExpression).then_ignore(end()),
    ))
}

// =====================================================================================================================
// AST Evaluation Helpers
// =====================================================================================================================

// =====================================================================================================================
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
        // Bytes comparison (only == and !=)
        (Value::Bytes(l), Value::Bytes(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            _ => return Err("Bytes comparison only supports == and !=".into()),
        }),
        // List comparison (only == and !=)
        (Value::List(l), Value::List(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            _ => return Err("List comparison only supports == and !=".into()),
        }),
        // Map comparison (only == and !=)
        (Value::Map(l), Value::Map(r)) => Ok(match op {
            CompOp::Eq => l == r,
            CompOp::NotEq => l != r,
            _ => return Err("Map comparison only supports == and !=".into()),
        }),
        // Different types: only == and != are valid (always false/true respectively)
        _ => Ok(match op {
            CompOp::Eq => false,
            CompOp::NotEq => true,
            _ => {
                return Err(format!(
                    "Cannot compare different types with {:?}: left={:?}, right={:?}",
                    op,
                    std::mem::discriminant(left),
                    std::mem::discriminant(right)
                )
                .into())
            }
        }),
    }
}

// =====================================================================================================================
/// Evaluate a pre-resolved path - NO runtime lookup needed!
#[inline]
fn evaluate_resolved_path(path: &ResolvedPath, ctx: &EvalContext) -> Result<Value> {
    // Direct call to pre-resolved accessor - no lookup, no clone!
    let value = path.accessor.get(ctx, &path.full_path)?;

    // If no indexes, return directly without any allocation
    if path.indexes.is_empty() {
        return Ok(value);
    }

    // Apply indexes
    let mut current = value;
    for index in &path.indexes {
        current = apply_index(&current, index)?;
    }

    Ok(current)
}

// =====================================================================================================================
/// Apply an index to a value
/// AZH: perhaps later index resolution we shall be delegated to integrator ...
/// perf. tests shall be used to compare different options.
/// For the time being the optimization without certitude that it worth it is postponed.
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
                .map(|c| Value::string(c.to_string()))
                .ok_or_else(|| format!("Index {} out of bounds", i).into())
        }
        _ => Err(format!("Cannot index {:?} with {:?}", value, index).into()),
    }
}

// =====================================================================================================================
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
            Ok(Value::string(format!("{}{}", l, r)))
        }
        _ => Err(format!(
            "Cannot perform math operation on {:?} and {:?}",
            std::mem::discriminant(left),
            std::mem::discriminant(right)
        )
        .into()),
    }
}

// =====================================================================================================================
// Arena-based AST Evaluation (cache-friendly traversal, NO runtime path lookup!)
// =====================================================================================================================

/// Zero-allocation argument evaluator for function calls.
/// Implements lazy evaluation - arguments are only evaluated when requested.
struct ArenaArgs<'a> {
    arena: &'a AstArena,
    args: &'a [ArenaArgExpr],
    ctx: *mut EvalContext, // Raw pointer for interior mutability
}

impl<'a> ArenaArgs<'a> {
    #[inline]
    fn new(arena: &'a AstArena, args: &'a [ArenaArgExpr], ctx: &mut EvalContext) -> Self {
        Self {
            arena,
            args,
            ctx: ctx as *mut EvalContext,
        }
    }
}

impl<'a> Args for ArenaArgs<'a> {
    #[inline]
    fn ctx(&mut self) -> &mut EvalContext {
        // SAFETY: We have exclusive access through the mutable reference in `new`,
        // and ArenaArgs is never shared or cloned.
        unsafe { &mut *self.ctx }
    }

    #[inline]
    fn len(&self) -> usize {
        self.args.len()
    }

    #[inline]
    fn get(&mut self, index: usize) -> Result<Value> {
        if index >= self.args.len() {
            return Err(format!("Argument index {} out of bounds", index).into());
        }

        // SAFETY: We have exclusive access, same as ctx()
        let ctx = unsafe { &mut *self.ctx };

        match &self.args[index] {
            ArenaArgExpr::Positional(value_ref) => {
                arena_evaluate_value_expr(*value_ref, self.arena, ctx)
            }
            ArenaArgExpr::Named { value, .. } => arena_evaluate_value_expr(*value, self.arena, ctx),
        }
    }

    #[inline]
    fn name(&self, index: usize) -> Option<&str> {
        if index >= self.args.len() {
            return None;
        }
        match &self.args[index] {
            ArenaArgExpr::Positional(_) => None,
            ArenaArgExpr::Named { name, .. } => Some(name),
        }
    }
}

/// Evaluate the arena-based root expression
/// Note: resolver is NOT used at runtime - paths are pre-resolved at parse time!
#[inline]
fn arena_evaluate_root(
    root: &ArenaRootExpr,
    arena: &AstArena,
    ctx: &mut EvalContext,
) -> Result<Value> {
    match root {
        ArenaRootExpr::EditorStatement(stmt) => {
            let should_execute = if let Some(cond_ref) = stmt.condition {
                arena_evaluate_bool_expr(cond_ref, arena, ctx)?
            } else {
                true
            };

            if should_execute {
                arena_evaluate_function_call(stmt.editor, arena, ctx)?;
            }

            Ok(Value::Nil)
        }
        ArenaRootExpr::BooleanExpression(expr_ref) => {
            let result = arena_evaluate_bool_expr(*expr_ref, arena, ctx)?;
            Ok(Value::Bool(result))
        }
        ArenaRootExpr::MathExpression(expr_ref) => arena_evaluate_math_expr(*expr_ref, arena, ctx),
    }
}

/// Evaluate an arena-based boolean expression
#[inline]
fn arena_evaluate_bool_expr(
    expr_ref: BoolExprRef,
    arena: &AstArena,
    ctx: &mut EvalContext,
) -> Result<bool> {
    match arena.get_bool(expr_ref) {
        ArenaBoolExpr::Literal(b) => Ok(*b),
        ArenaBoolExpr::Comparison { left, op, right } => {
            let left_val = arena_evaluate_value_expr(*left, arena, ctx)?;
            let right_val = arena_evaluate_value_expr(*right, arena, ctx)?;
            evaluate_comparison(&left_val, op, &right_val)
        }
        ArenaBoolExpr::Converter(fc_ref) => {
            let result = arena_evaluate_function_call(*fc_ref, arena, ctx)?;
            match result {
                Value::Bool(b) => Ok(b),
                _ => Err("Converter did not return a boolean".into()),
            }
        }
        ArenaBoolExpr::Path(resolved_path) => {
            // Use pre-resolved path - NO runtime lookup!
            let value = evaluate_resolved_path(resolved_path, ctx)?;
            match value {
                Value::Bool(b) => Ok(b),
                _ => Err("Path did not return a boolean".into()),
            }
        }
        ArenaBoolExpr::Not(inner_ref) => {
            let result = arena_evaluate_bool_expr(*inner_ref, arena, ctx)?;
            Ok(!result)
        }
        ArenaBoolExpr::And(left_ref, right_ref) => {
            let left_result = arena_evaluate_bool_expr(*left_ref, arena, ctx)?;
            if !left_result {
                return Ok(false);
            }
            arena_evaluate_bool_expr(*right_ref, arena, ctx)
        }
        ArenaBoolExpr::Or(left_ref, right_ref) => {
            let left_result = arena_evaluate_bool_expr(*left_ref, arena, ctx)?;
            if left_result {
                return Ok(true);
            }
            arena_evaluate_bool_expr(*right_ref, arena, ctx)
        }
    }
}

/// Evaluate an arena-based value expression
#[inline]
fn arena_evaluate_value_expr(
    expr_ref: ValueExprRef,
    arena: &AstArena,
    ctx: &mut EvalContext,
) -> Result<Value> {
    match arena.get_value(expr_ref) {
        ArenaValueExpr::Literal(v) => Ok(v.clone()),
        ArenaValueExpr::Path(resolved_path) => {
            // Use pre-resolved path - NO runtime lookup!
            evaluate_resolved_path(resolved_path, ctx)
        }
        ArenaValueExpr::List(items) => {
            let values: Result<Vec<Value>> = items
                .iter()
                .map(|item_ref| arena_evaluate_value_expr(*item_ref, arena, ctx))
                .collect();
            Ok(Value::List(values?))
        }
        ArenaValueExpr::Map(entries) => {
            let mut map = std::collections::HashMap::new();
            for (key, value_ref) in entries {
                let value = arena_evaluate_value_expr(*value_ref, arena, ctx)?;
                map.insert(key.clone(), value);
            }
            Ok(Value::Map(map))
        }
        ArenaValueExpr::FunctionCall(fc_ref) => arena_evaluate_function_call(*fc_ref, arena, ctx),
        ArenaValueExpr::Math(math_ref) => arena_evaluate_math_expr(*math_ref, arena, ctx),
    }
}

/// Evaluate an arena-based function call  
/// ZERO ALLOCATION - uses lazy ArenaArgs for argument evaluation
#[inline]
fn arena_evaluate_function_call(
    fc_ref: FunctionCallRef,
    arena: &AstArena,
    ctx: &mut EvalContext,
) -> Result<Value> {
    let fc = arena.get_func(fc_ref);

    let callback = fc
        .callback
        .as_ref()
        .ok_or_else(|| -> BoxError { format!("Unknown function: {}", fc.name).into() })?;

    // ZERO ALLOCATION: Create lazy args evaluator
    let mut args = ArenaArgs::new(arena, &fc.args, ctx);

    // Callback uses lazy evaluation - only requested args are computed
    let result = callback(&mut args)?;

    let mut current = result;
    for index in &fc.indexes {
        current = apply_index(&current, index)?;
    }

    Ok(current)
}

/// Evaluate an arena-based math expression
#[inline]
fn arena_evaluate_math_expr(
    expr_ref: MathExprRef,
    arena: &AstArena,
    ctx: &mut EvalContext,
) -> Result<Value> {
    match arena.get_math(expr_ref) {
        ArenaMathExpr::Primary(value_ref) => arena_evaluate_value_expr(*value_ref, arena, ctx),
        ArenaMathExpr::Negate(inner_ref) => {
            let value = arena_evaluate_math_expr(*inner_ref, arena, ctx)?;
            match value {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err("Cannot negate non-numeric value".into()),
            }
        }
        ArenaMathExpr::Binary {
            left: left_ref,
            op,
            right: right_ref,
        } => {
            let left_val = arena_evaluate_math_expr(*left_ref, arena, ctx)?;
            let right_val = arena_evaluate_math_expr(*right_ref, arena, ctx)?;
            evaluate_math_op(&left_val, op, &right_val)
        }
    }
}
