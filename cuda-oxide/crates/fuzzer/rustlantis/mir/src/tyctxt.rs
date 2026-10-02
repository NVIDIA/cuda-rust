use config::TyConfig;
use std::{collections::HashMap, slice};

use index_vec::IndexVec;

use crate::{
    serialize::Serialize,
    syntax::{Adt, TyId, TyKind},
};

#[derive(Debug, Clone, Copy)]
pub struct AdtMeta {
    pub copy: bool,
}

impl AdtMeta {
    fn derive_attrs(&self) -> String {
        let mut attrs = vec!["Debug"];
        if self.copy {
            attrs.push("Copy");
            attrs.push("Clone");
        }

        let list: String = attrs.iter().intersperse(&",").copied().collect();
        if list.is_empty() {
            "".to_owned()
        } else {
            format!("#[derive({list})]\n")
        }
    }
}

pub struct TyCtxt {
    tys: IndexVec<TyId, TyKind>,
    adt_meta: HashMap<TyId, AdtMeta>,
    pub config: TyConfig,
}

impl TyCtxt {
    pub const UNIT: TyId = TyId::from_usize_unchecked(0);
    pub const BOOL: TyId = TyId::from_usize_unchecked(1);
    pub const CHAR: TyId = TyId::from_usize_unchecked(2);
    pub const ISIZE: TyId = TyId::from_usize_unchecked(3);
    pub const I8: TyId = TyId::from_usize_unchecked(4);
    pub const I16: TyId = TyId::from_usize_unchecked(5);
    pub const I32: TyId = TyId::from_usize_unchecked(6);
    pub const I64: TyId = TyId::from_usize_unchecked(7);
    pub const I128: TyId = TyId::from_usize_unchecked(8);
    pub const USIZE: TyId = TyId::from_usize_unchecked(9);
    pub const U8: TyId = TyId::from_usize_unchecked(10);
    pub const U16: TyId = TyId::from_usize_unchecked(11);
    pub const U32: TyId = TyId::from_usize_unchecked(12);
    pub const U64: TyId = TyId::from_usize_unchecked(13);
    pub const U128: TyId = TyId::from_usize_unchecked(14);
    pub const F32: TyId = TyId::from_usize_unchecked(15);
    pub const F64: TyId = TyId::from_usize_unchecked(16);
    pub const INTEGER_TYPES: [TyId; 12] = [
        Self::ISIZE,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::I128,
        Self::USIZE,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::U128,
    ];

    pub fn from_primitives(config: TyConfig) -> Self {
        let primitives: [TyKind; 17] = [
            TyKind::Unit,
            TyKind::Bool,
            TyKind::Char,
            TyKind::ISIZE,
            TyKind::I8,
            TyKind::I16,
            TyKind::I32,
            TyKind::I64,
            TyKind::I128,
            TyKind::USIZE,
            TyKind::U8,
            TyKind::U16,
            TyKind::U32,
            TyKind::U64,
            TyKind::U128,
            TyKind::F32,
            TyKind::F64,
        ];
        let tys = IndexVec::from_iter(primitives);
        Self {
            tys,
            adt_meta: HashMap::new(),
            config,
        }
    }

    pub fn push(&mut self, kind: TyKind) -> TyId {
        assert!(kind.is_structural());
        self.tys.push(kind)
    }

    pub fn push_adt(&mut self, adt: Adt, meta: AdtMeta) -> TyId {
        let id = self.tys.push(TyKind::Adt(adt));
        self.adt_meta.insert(id, meta);
        id
    }

    pub fn meta(&self, ty: TyId) -> AdtMeta {
        self.adt_meta[&ty]
    }

    pub fn kind(&self, ty: TyId) -> &TyKind {
        &self.tys[ty]
    }

    pub fn indices(&self) -> impl Iterator<Item = TyId> {
        self.tys.indices()
    }

    pub fn iter(&self) -> slice::Iter<'_, TyKind> {
        self.tys.iter()
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (TyId, &TyKind)> {
        self.tys.iter_enumerated()
    }

    pub fn len(&self) -> usize {
        self.tys.len()
    }

    pub fn serialize(&self) -> String {
        let mut str = String::new();
        for (id, adt) in self.tys.iter_enumerated().filter(|(_, kind)| kind.is_adt()) {
            let TyKind::Adt(adt) = adt else {
                panic!("not an adt");
            };
            str += &self.adt_meta[&id].derive_attrs();
            if adt.is_enum() {
                let variants: String = adt
                    .variants
                    .iter_enumerated()
                    .map(|(vid, def)| {
                        format!("{}{{\n{}\n}}", vid.identifier(), def.serialize(self))
                    })
                    .intersperse(",\n".to_string())
                    .collect();
                str += &format!("pub enum {} {{\n{variants}}}\n", id.type_name())
            } else {
                let def = adt.variants.first().expect("has only one variant");
                str += &format!(
                    "pub struct {} {{\n{}}}\n",
                    id.type_name(),
                    def.serialize(self)
                )
            }
        }
        str
    }
}
