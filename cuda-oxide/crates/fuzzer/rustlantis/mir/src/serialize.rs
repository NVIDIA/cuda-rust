use crate::{syntax::*, tyctxt::TyCtxt};

#[derive(Debug, Clone, Copy)]
pub enum CallSynatx {
    /// Call(bb2, _1, fn2(_3))
    /// Up to 2023-08-19
    V1,
    /// Call(_1 = fn2(_3), bb2)
    /// Up to 2023-11-14
    V2,
    /// Call(_1 = fn2(_3), bb2, UnwindUnreachable())
    /// Uo to 2023-12-26
    V3,
    /// Call(_1 = fn2(_3), ReturnTo(bb2), UnwindUnreachable())
    /// Current
    V4,
}

impl From<&str> for CallSynatx {
    fn from(value: &str) -> Self {
        match value {
            "v1" => Self::V1,
            "v2" => Self::V2,
            "v3" => Self::V3,
            "v4" => Self::V4,
            _ => panic!("invalid syntax version {value}"),
        }
    }
}

pub trait Serialize {
    fn serialize(&self, tcx: &TyCtxt) -> String;
}

impl Serialize for TyId {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        match self.kind(tcx) {
            TyKind::Unit => "()".to_owned(),
            TyKind::Bool => "bool".to_owned(),
            TyKind::Char => "char".to_owned(),
            TyKind::Int(IntTy::Isize) => "isize".to_owned(),
            TyKind::Int(IntTy::I8) => "i8".to_owned(),
            TyKind::Int(IntTy::I16) => "i16".to_owned(),
            TyKind::Int(IntTy::I32) => "i32".to_owned(),
            TyKind::Int(IntTy::I64) => "i64".to_owned(),
            TyKind::Int(IntTy::I128) => "i128".to_owned(),

            TyKind::Uint(UintTy::Usize) => "usize".to_owned(),
            TyKind::Uint(UintTy::U8) => "u8".to_owned(),
            TyKind::Uint(UintTy::U16) => "u16".to_owned(),
            TyKind::Uint(UintTy::U32) => "u32".to_owned(),
            TyKind::Uint(UintTy::U64) => "u64".to_owned(),
            TyKind::Uint(UintTy::U128) => "u128".to_owned(),

            TyKind::Float(FloatTy::F32) => "f32".to_owned(),
            TyKind::Float(FloatTy::F64) => "f64".to_owned(),
            // Pointer types
            TyKind::RawPtr(ty, mutability) => {
                format!("{}{}", mutability.ptr_prefix_str(), ty.serialize(tcx))
            }
            TyKind::Ref(ty, mutability) => {
                format!("&'static {}{}", mutability.prefix_str(), ty.serialize(tcx))
            }
            // Sequence types
            TyKind::Tuple(elems) => {
                if elems.len() == 1 {
                    format!("({},)", elems[0].serialize(tcx))
                } else {
                    format!("({})", elems.as_slice().serialize(tcx))
                }
            }
            TyKind::Array(ty, len) => {
                format!("[{}; {len}]", ty.serialize(tcx))
            }
            // User-defined type
            TyKind::Adt(_) => self.type_name(),
        }
    }
}

impl Serialize for &[TyId] {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        self.iter()
            .map(|ty| ty.serialize(tcx))
            .intersperse(", ".to_string())
            .collect()
    }
}

impl Place {
    pub fn serialize_value(&self, tcx: &TyCtxt) -> String {
        let str = self.local().identifier();
        self.projection().iter().fold(str, |acc, proj| match proj {
            ProjectionElem::Deref => format!("(*{acc})"),
            ProjectionElem::TupleField(id) => format!("{acc}.{}", id.index()),
            ProjectionElem::Field(id) => format!("{acc}.{}", id.identifier()),
            ProjectionElem::Index(local) => format!("{acc}[{}]", local.identifier()),
            ProjectionElem::DowncastField(vid, fid, ty) => format!(
                "Field::<{}>(Variant({acc}, {}), {})",
                ty.serialize(tcx),
                vid.index(),
                fid.index()
            ),
            ProjectionElem::ConstantIndex { offset } => format!("{acc}[{offset}]"),
        })
    }

    // Place context needs place!() for Field(Variant())
    pub fn serialize_place(&self, tcx: &TyCtxt) -> String {
        let str = self.local().identifier();
        self.projection().iter().fold(str, |acc, proj| match proj {
            ProjectionElem::Deref => format!("(*{acc})"),
            ProjectionElem::TupleField(id) => format!("{acc}.{}", id.index()),
            ProjectionElem::Field(id) => format!("{acc}.{}", id.identifier()),
            ProjectionElem::Index(local) => format!("{acc}[{}]", local.identifier()),
            ProjectionElem::DowncastField(vid, fid, ty) => format!(
                "place!(Field::<{}>(Variant({acc}, {}), {}))",
                ty.serialize(tcx),
                vid.index(),
                fid.index()
            ),
            ProjectionElem::ConstantIndex { offset } => format!("{acc}[{offset}]"),
        })
    }
}

impl Serialize for Literal {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        match self {
            Literal::Uint(i, _) => format!("{i}_{}", self.ty().serialize(tcx)),
            Literal::Int(i, _) if *i < 0 => format!("({i}_{})", self.ty().serialize(tcx)),
            Literal::Int(i, _) => format!("{i}_{}", self.ty().serialize(tcx)),
            Literal::Float(f, _) => {
                if f.is_nan() {
                    format!("{}::NAN", self.ty().serialize(tcx))
                } else if f.is_infinite() {
                    if f.is_sign_positive() {
                        format!("{}::INFINITY", self.ty().serialize(tcx))
                    } else {
                        format!("{}::NEG_INFINITY", self.ty().serialize(tcx))
                    }
                } else if *f < 0. {
                    format!("({f}_{})", self.ty().serialize(tcx))
                } else {
                    format!("{f}_{}", self.ty().serialize(tcx))
                }
            }
            Literal::Bool(b) => b.to_string(),
            Literal::Char(c) => format!("'\\u{{{:x}}}'", u32::from(*c)),
        }
    }
}

impl Serialize for Operand {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        match self {
            Operand::Copy(place) => place.serialize_value(tcx),
            Operand::Move(place) => format!("Move({})", place.serialize_value(tcx)),
            Operand::Constant(lit) => lit.serialize(tcx),
            Operand::Static(id, _) => format!("Static({})", id.identifier()),
            Operand::StaticMut(id, _) => format!("StaticMut({})", id.identifier()),
        }
    }
}

impl Serialize for Rvalue {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        match self {
            Rvalue::Use(a) => a.serialize(tcx),
            Rvalue::UnaryOp(op, a) => format!("{}{}", op.symbol(), a.serialize(tcx)),
            Rvalue::BinaryOp(BinOp::Offset, a, b) => {
                format!("Offset({}, {})", a.serialize(tcx), b.serialize(tcx))
            }
            Rvalue::BinaryOp(op, a, b) => {
                format!("{} {} {}", a.serialize(tcx), op.symbol(), b.serialize(tcx))
            }
            Rvalue::CheckedBinaryOp(op, a, b) => format!(
                "Checked({} {} {})",
                a.serialize(tcx),
                op.symbol(),
                b.serialize(tcx)
            ),

            Rvalue::Cast(a, target) => format!("{} as {}", a.serialize(tcx), target.serialize(tcx)),
            Rvalue::Len(place) => format!("Len({})", place.serialize_value(tcx)),
            Rvalue::Discriminant(place) => format!("Discriminant({})", place.serialize_value(tcx)),
            Rvalue::AddressOf(Mutability::Not, place) => {
                format!("core::ptr::addr_of!({})", place.serialize_place(tcx))
            }
            Rvalue::AddressOf(Mutability::Mut, place) => {
                format!("core::ptr::addr_of_mut!({})", place.serialize_place(tcx))
            }
            Rvalue::Ref(Mutability::Not, place) => {
                format!("&{}", place.serialize_place(tcx))
            }
            Rvalue::Ref(Mutability::Mut, place) => {
                format!("&mut {}", place.serialize_place(tcx))
            }
            Rvalue::Aggregate(kind, operands) => match kind {
                AggregateKind::Array(_) => {
                    let list: String = operands
                        .iter()
                        .map(|op| op.serialize(tcx))
                        .intersperse(",".to_owned())
                        .collect();
                    format!("[{list}]")
                }
                AggregateKind::Tuple => match operands.len() {
                    0 => "()".to_owned(),
                    1 => format!("({},)", operands.first().unwrap().serialize(tcx)),
                    _ => format!(
                        "({})",
                        operands
                            .iter()
                            .map(|l| l.serialize(tcx))
                            .intersperse(", ".to_owned())
                            .collect::<String>()
                    ),
                },
                AggregateKind::Adt(ty, variant) => {
                    let TyKind::Adt(adt) = ty.kind(tcx) else {
                        panic!("not an adt");
                    };
                    let list: String = operands
                        .iter_enumerated()
                        .map(|(fid, op)| format!("{}: {}", fid.identifier(), op.serialize(tcx)))
                        .intersperse(",".to_owned())
                        .collect();
                    if adt.is_enum() {
                        format!("{}::{} {{ {list} }}", ty.type_name(), variant.identifier())
                    } else {
                        format!("{} {{ {list} }}", ty.type_name())
                    }
                }
            },
        }
    }
}

impl Serialize for Statement {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        match self {
            Statement::Assign(place, rvalue) => {
                format!("{} = {}", place.serialize_place(tcx), rvalue.serialize(tcx))
            }
            Statement::StorageLive(local) => format!("StorageLive({})", local.identifier()),
            Statement::StorageDead(local) => format!("StorageDead({})", local.identifier()),
            Statement::Deinit(local) => format!("Deinit({})", local.serialize_value(tcx)),
            Statement::SetDiscriminant(place, discr) => {
                format!("SetDiscriminant({}, {discr})", place.serialize_value(tcx))
            }
            Statement::Retag(place) => format!("Retag({})", place.serialize_value(tcx)),
            Statement::Nop => String::default(),
        }
    }
}

impl Terminator {
    fn serialize(&self, tcx: &TyCtxt, call_syntax: CallSynatx) -> String {
        match self {
            Terminator::Return => "Return()".to_owned(),
            Terminator::Goto { target } => format!("Goto({})", target.identifier()),
            Terminator::Unreachable => "Unreachable()".to_owned(),
            Terminator::Drop { place, target } => {
                format!(
                    "Drop({}, {})",
                    place.serialize_value(tcx),
                    target.identifier()
                )
            }
            Terminator::Call {
                destination,
                target,
                callee,
                args,
            } => {
                let args_list: String = args
                    .iter()
                    .map(|arg| arg.serialize(tcx))
                    .intersperse(", ".to_owned())
                    .collect();
                let fn_name = match callee {
                    Callee::Generated(func) => func.identifier(),
                    Callee::Named(func) => func.to_string(),
                    Callee::Intrinsic(func) => format!("core::intrinsics::{func}"),
                };
                match call_syntax {
                    CallSynatx::V1 => format!(
                        "Call({}, {}, {fn_name}({args_list}))",
                        destination.serialize_place(tcx),
                        target.identifier(),
                    ),
                    CallSynatx::V2 => format!(
                        "Call({} = {fn_name}({args_list}), {})",
                        destination.serialize_place(tcx),
                        target.identifier(),
                    ),
                    CallSynatx::V3 => format!(
                        "Call({} = {fn_name}({args_list}), {}, UnwindUnreachable())",
                        destination.serialize_place(tcx),
                        target.identifier(),
                    ),
                    CallSynatx::V4 => format!(
                        "Call({} = {fn_name}({args_list}), ReturnTo({}), UnwindUnreachable())",
                        destination.serialize_place(tcx),
                        target.identifier(),
                    ),
                }
            }
            Terminator::SwitchInt { discr, targets } => {
                let arms = targets.match_arms();
                format!("match {} {{\n{}\n}}", discr.serialize(tcx), arms)
            }
            Terminator::Hole => unreachable!("hole"),
        }
    }
}

impl BasicBlockData {
    fn serialize(&self, tcx: &TyCtxt, call_syntax: CallSynatx) -> String {
        let mut stmts: String = self
            .statements
            .iter()
            .filter(|stmt| !matches!(stmt, Statement::Nop))
            .map(|stmt| format!("{};\n", stmt.serialize(tcx)))
            .collect();
        stmts.push_str(&self.terminator.serialize(tcx, call_syntax));
        stmts
    }
}

impl Serialize for VariantDef {
    fn serialize(&self, tcx: &TyCtxt) -> String {
        self.fields
            .iter_enumerated()
            .map(|(id, ty)| format!("{}: {},\n", id.identifier(), ty.serialize(tcx)))
            .collect()
    }
}

impl Body {
    fn serialize(&self, tcx: &TyCtxt, call_syntax: CallSynatx) -> String {
        // Return type annotation
        let mut body: String = format!("type RET = {};\n", self.return_ty().serialize(tcx));
        // Declarations
        body.extend(self.vars_iter().map(|idx| {
            let decl = &self.local_decls[idx];
            format!("let {}: {};\n", idx.identifier(), decl.ty.serialize(tcx))
        }));
        let mut bbs = self.basic_blocks.iter_enumerated();
        // First bb
        body.push_str(&format!(
            "{{\n{}\n}}\n",
            bbs.next()
                .expect("body contains at least one bb")
                .1
                .serialize(tcx, call_syntax)
        ));
        // Other bbs
        body.extend(bbs.map(|(idx, bb)| {
            format!(
                "{} = {{\n{}\n}}\n",
                idx.identifier(),
                bb.serialize(tcx, call_syntax)
            )
        }));
        format!("mir! {{\n{body}\n}}")
    }
}

impl Program {
    pub fn serialize(&self, tcx: &TyCtxt, call_syntax: CallSynatx) -> String {
        let mut program = Program::HEADER.to_string();

        if self.use_debug_dumper {
            program += Program::DEBUG_DUMPER;
        } else {
            program += Program::DUMPER;
        }

        program.extend(self.statics.iter_enumerated().map(|(idx, decl)| {
            let keyword = match decl.mutability {
                Mutability::Not => "static",
                Mutability::Mut => "static mut",
            };
            format!(
                "{keyword} {}: {} = {};\n",
                idx.identifier(),
                decl.ty.serialize(tcx),
                decl.init.serialize(tcx)
            )
        }));

        program.extend(self.functions.iter_enumerated().map(|(idx, body)| {
            let args_list: String = body
                .args_iter()
                .map(|arg| {
                    let decl = &body.local_decls[arg];
                    format!(
                        "{}{}: {}",
                        decl.mutability.prefix_str(),
                        arg.identifier(),
                        decl.ty.serialize(tcx)
                    )
                })
                .intersperse(",".to_string())
                .collect();
            format!(
                "{}\n{}fn {}({}) -> {} {{\n{}\n}}\n",
                Program::FUNCTION_ATTRIBUTE,
                if body.public { "pub " } else { "" },
                idx.identifier(),
                args_list,
                body.return_ty().serialize(tcx),
                body.serialize(tcx, call_syntax)
            )
        }));
        let arg_list: String = self
            .entry_args
            .iter()
            .map(|arg| format!("std::hint::black_box({})", arg.serialize(tcx)))
            .intersperse(", ".to_string())
            .collect();

        let first_fn: String = self
            .functions
            .indices()
            .next()
            .expect("program has functions")
            .identifier();

        let hash_printer = if self.use_debug_dumper {
            ""
        } else {
            r#"
                unsafe {
                    println!("hash: {}", H.finish());
                }
            "#
        };

        program.push_str(&format!(
            "pub fn main() {{
                {first_fn}({arg_list});
                {hash_printer}
            }}"
        ));
        program
    }
}

#[cfg(test)]
mod tests {
    use super::{CallSynatx, Serialize};
    use crate::{syntax::*, tyctxt::TyCtxt};
    use config::TyConfig;

    #[test]
    fn serialize_body() {
        let mut body = Body::new(&[TyCtxt::BOOL], TyCtxt::BOOL, true);
        let statements = vec![Statement::Assign(
            Place::RETURN_SLOT,
            Rvalue::UnaryOp(UnOp::Not, Operand::Copy(Place::from_local(Local::new(1)))),
        )];
        body.new_basic_block(BasicBlockData {
            statements,
            terminator: Terminator::Return,
        });
    }

    #[test]
    fn serialize_literal() {
        let tcx = TyCtxt::from_primitives(TyConfig::default());
        let nan = Literal::Float(f32::NAN as f64, FloatTy::F32);
        assert_eq!(nan.serialize(&tcx), "f32::NAN");

        let inf = Literal::Float(f32::INFINITY as f64, FloatTy::F32);
        assert_eq!(inf.serialize(&tcx), "f32::INFINITY");
    }

    #[test]
    fn serialize_static_operands() {
        let mut tcx = TyCtxt::from_primitives(TyConfig::default());
        let shared_ref = tcx.push(TyKind::Ref(TyCtxt::I32, Mutability::Not));
        let mut_ptr = tcx.push(TyKind::RawPtr(TyCtxt::I64, Mutability::Mut));
        let mut program = Program::new(false);
        let immutable = program.push_static(StaticDecl {
            ty: TyCtxt::I32,
            mutability: Mutability::Not,
            init: 1_i32.into(),
        });
        let mutable = program.push_static(StaticDecl {
            ty: TyCtxt::I64,
            mutability: Mutability::Mut,
            init: 2_i64.into(),
        });

        assert_eq!(
            Operand::Static(immutable, shared_ref).serialize(&tcx),
            "Static(static0)"
        );
        assert_eq!(
            Operand::StaticMut(mutable, mut_ptr).serialize(&tcx),
            "StaticMut(static1)"
        );

        let mut body = Body::new(&[], TyCtxt::UNIT, true);
        body.new_basic_block(BasicBlockData {
            statements: vec![],
            terminator: Terminator::Return,
        });
        program.push_fn(body);
        let source = program.serialize(&tcx, CallSynatx::V4);
        assert!(source.contains("static static0: i32 = 1_i32;"));
        assert!(source.contains("static mut static1: i64 = 2_i64;"));
    }
}
