//! Arena-based AST storage and conversion from boxed AST
//!
//! Provides cache-friendly AST storage and compile-time optimizations like:
//! - Path resolution at parse time
//! - Constant folding for literals

use crate::{PathResolver, Result, Value};

use super::ast::*;

// =====================================================================================================================
// Arena Storage
// =====================================================================================================================

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

// =====================================================================================================================
// AST to Arena Conversion
// =====================================================================================================================

/// Helper to resolve a PathExpr to ResolvedPath at parse time
fn resolve_path(path: &PathExpr, resolver: &PathResolver) -> Result<ResolvedPath> {
    let full_path = path.segments.join(".");
    let accessor = resolver()?;
    Ok(ResolvedPath {
        full_path,
        accessor,
        indexes: path.indexes.clone(),
    })
}

/// Convert boxed AST to arena-based AST for cache-friendly execution
/// Path resolution happens HERE (once at parse time), not at each execution!
pub fn convert_to_arena(
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

// =====================================================================================================================
// Constant Folding Helpers
// =====================================================================================================================

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

/// Try to get literal value from a MathExpr
fn try_get_math_literal(expr: &MathExpr) -> Option<&Value> {
    match expr {
        MathExpr::Primary(ValueExpr::Literal(v)) => Some(v),
        _ => None,
    }
}

// Use shared operations from ops module
use super::ops::{compare, try_math_op};

// =====================================================================================================================
// Conversion Functions
// =====================================================================================================================

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
                if let Some(result) = compare(left_val, op, right_val).ok() {
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
                return convert_bool_expr(right, arena, resolver);
            }
            let right_ref = convert_bool_expr(right, arena, resolver)?;
            // CONSTANT FOLDING: x && false => false, x && true => x
            if let Some(right_val) = arena_bool_is_literal(arena, right_ref) {
                if !right_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(false)));
                }
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
                return convert_bool_expr(right, arena, resolver);
            }
            let right_ref = convert_bool_expr(right, arena, resolver)?;
            // CONSTANT FOLDING: x || true => true, x || false => x
            if let Some(right_val) = arena_bool_is_literal(arena, right_ref) {
                if right_val {
                    return Ok(arena.alloc_bool(ArenaBoolExpr::Literal(true)));
                }
                return Ok(left_ref);
            }
            ArenaBoolExpr::Or(left_ref, right_ref)
        }
    };
    Ok(arena.alloc_bool(arena_expr))
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
                if let Some(result) = try_math_op(left_val, op, right_val) {
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
