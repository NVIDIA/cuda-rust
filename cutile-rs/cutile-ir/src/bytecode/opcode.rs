/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Frozen opcode assignments for all CUDA Tile operations.
//!
//! Ported from `BytecodeOpcodes.td`. These values must never be renumbered
//! for backward compatibility.

/// Bytecode opcode for a single CUDA Tile operation.
///
/// Public operations occupy the range `0x000 ..= 0xFFF` and public-preview
/// operations the range `0x4000 ..= 0x4FFF`. Each variant's discriminant is
/// the on-wire opcode value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Opcode {
    AbsF = 0x00,
    AbsI = 0x01,
    AddF = 0x02,
    AddI = 0x03,
    AndI = 0x04,
    Assert = 0x05,
    Assume = 0x06,
    AtomicCAS = 0x07,
    AtomicRMW = 0x08,
    Bitcast = 0x09,
    Break = 0x0A,
    Broadcast = 0x0B,
    Cat = 0x0C,
    Ceil = 0x0D,
    CmpF = 0x0E,
    CmpI = 0x0F,
    Constant = 0x10,
    Continue = 0x11,
    Cos = 0x12,
    CosH = 0x13,
    DivF = 0x14,
    DivI = 0x15,
    Entry = 0x16,
    Exp = 0x17,
    Exp2 = 0x18,
    ExtI = 0x25,
    Extract = 0x26,
    Floor = 0x27,
    Fma = 0x28,
    For = 0x29,
    FToF = 0x2A,
    FToI = 0x2B,
    GetGlobal = 0x2C,
    GetIndexSpaceShape = 0x2D,
    GetNumTileBlocks = 0x2E,
    GetTensorShape = 0x2F,
    GetTileBlockId = 0x30,
    Global = 0x31,
    If = 0x32,
    IntToPtr = 0x33,
    Iota = 0x3A,
    IToF = 0x3B,
    JoinTokens = 0x3C,
    LoadPtrTko = 0x3D,
    LoadViewTko = 0x3E,
    Log = 0x3F,
    Log2 = 0x40,
    Loop = 0x41,
    MakePartitionView = 0x42,
    MakeTensorView = 0x43,
    MakeToken = 0x44,
    MaxF = 0x45,
    MaxI = 0x46,
    MinF = 0x47,
    MinI = 0x48,
    MmaF = 0x49,
    MmaI = 0x4A,
    Module = 0x4B,
    MulF = 0x4C,
    MulhiI = 0x4D,
    MulI = 0x4E,
    NegF = 0x4F,
    NegI = 0x50,
    Offset = 0x51,
    OrI = 0x52,
    Permute = 0x53,
    Pow = 0x54,
    Print = 0x55,
    PtrToInt = 0x56,
    PtrToPtr = 0x57,
    Reduce = 0x58,
    RemF = 0x59,
    RemI = 0x5A,
    Reshape = 0x5B,
    Return = 0x5C,
    Rsqrt = 0x5D,
    Scan = 0x5E,
    Select = 0x5F,
    ShLI = 0x60,
    ShRI = 0x61,
    Sin = 0x62,
    SinH = 0x63,
    Sqrt = 0x64,
    StorePtrTko = 0x65,
    StoreViewTko = 0x66,
    SubF = 0x67,
    SubI = 0x68,
    Tan = 0x69,
    TanH = 0x6A,
    TruncI = 0x6B,
    XOrI = 0x6C,
    Yield = 0x6D,
    Atan2 = 0x6E,
    Pack = 0x6F,
    Unpack = 0x70,
    Alloca = 0x71,
    MmaFScaled = 0x72,
    MakeGatherScatterView = 0x73,
    MakeStridedView = 0x74,
    AtomicRedViewTko = 0x75,
    Insert = 0x76,
    GdcLaunchDependentsTko = 0x77,
    GdcWaitTko = 0x78,
    FPowI = 0x79,
    MemoryFenceAliasTko = 0x7A,
    MultimemLoadReduceViewTko = 0x7B,
    MultimemReduceViewTko = 0x7C,
    MultimemStoreViewTko = 0x7D,
    /// `preview$call`: call to an extern function supplied by a tile library.
    PreviewCall = 0x4002,
}

impl Opcode {
    /// Unqualified dialect operation name.
    pub const fn name(self) -> &'static str {
        use Opcode::*;
        match self {
            MultimemLoadReduceViewTko => "multimem_load_reduce_view_tko",
            MultimemReduceViewTko => "multimem_reduce_view_tko",
            MultimemStoreViewTko => "multimem_store_view_tko",
            AbsF => "absf",
            AbsI => "absi",
            AddF => "addf",
            AddI => "addi",
            AndI => "andi",
            Assert => "assert",
            Assume => "assume",
            Alloca => "alloca",
            Atan2 => "atan2",
            AtomicCAS => "atomic_cas_tko",
            AtomicRedViewTko => "atomic_red_view_tko",
            AtomicRMW => "atomic_rmw_tko",
            Bitcast => "bitcast",
            Break => "break",
            Broadcast => "broadcast",
            Cat => "cat",
            Ceil => "ceil",
            CmpF => "cmpf",
            CmpI => "cmpi",
            Constant => "constant",
            Continue => "continue",
            Cos => "cos",
            CosH => "cosh",
            DivF => "divf",
            DivI => "divi",
            Entry => "entry",
            Exp => "exp",
            Exp2 => "exp2",
            ExtI => "exti",
            Extract => "extract",
            Floor => "floor",
            Fma => "fma",
            For => "for",
            FToF => "ftof",
            FToI => "ftoi",
            GetGlobal => "get_global",
            GetIndexSpaceShape => "get_index_space_shape",
            GetNumTileBlocks => "get_num_tile_blocks",
            GetTensorShape => "get_tensor_shape",
            GetTileBlockId => "get_tile_block_id",
            Global => "global",
            If => "if",
            IntToPtr => "int_to_ptr",
            Iota => "iota",
            IToF => "itof",
            JoinTokens => "join_tokens",
            LoadPtrTko => "load_ptr_tko",
            LoadViewTko => "load_view_tko",
            Log => "log",
            Log2 => "log2",
            Loop => "loop",
            MakeGatherScatterView => "make_gather_scatter_view",
            MakePartitionView => "make_partition_view",
            MakeStridedView => "make_strided_view",
            MakeTensorView => "make_tensor_view",
            MakeToken => "make_token",
            MaxF => "maxf",
            MaxI => "maxi",
            MinF => "minf",
            MinI => "mini",
            MmaF => "mmaf",
            MmaFScaled => "mmaf_scaled",
            MmaI => "mmai",
            Module => "module",
            MulF => "mulf",
            MulhiI => "mulhii",
            MulI => "muli",
            NegF => "negf",
            NegI => "negi",
            Offset => "offset",
            OrI => "ori",
            Pack => "pack",
            Permute => "permute",
            Pow => "pow",
            Print => "print_tko",
            PtrToInt => "ptr_to_int",
            PtrToPtr => "ptr_to_ptr",
            Reduce => "reduce",
            RemF => "remf",
            RemI => "remi",
            Reshape => "reshape",
            Return => "return",
            Rsqrt => "rsqrt",
            Scan => "scan",
            Select => "select",
            ShLI => "shli",
            ShRI => "shri",
            Sin => "sin",
            SinH => "sinh",
            Sqrt => "sqrt",
            StorePtrTko => "store_ptr_tko",
            StoreViewTko => "store_view_tko",
            SubF => "subf",
            SubI => "subi",
            Tan => "tan",
            TanH => "tanh",
            TruncI => "trunci",
            Unpack => "unpack",
            XOrI => "xori",
            Yield => "yield",
            Insert => "insert",
            GdcLaunchDependentsTko => "gdc_launch_dependents_tko",
            GdcWaitTko => "gdc_wait_tko",
            FPowI => "fpowi",
            MemoryFenceAliasTko => "memory_fence_alias_tko",
            PreviewCall => "preview$call",
        }
    }

    /// Whether this is a public-preview operation. A module containing one
    /// is written with the header preview bit at [`super::BytecodeVersion::V13_5`].
    pub const fn is_preview(self) -> bool {
        matches!(self, Opcode::PreviewCall)
    }

    /// Minimum supported wire version for this opcode. Operand types and
    /// nondefault attributes can impose a newer requirement of their own.
    pub const fn minimum_version(self) -> super::BytecodeVersion {
        crate::requirements::opcode_requirement(self).since
    }

    /// Return the raw u16 opcode value for bytecode emission.
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Construct an opcode from its raw u16 value (inverse of [`as_u16`](Self::as_u16)).
    pub fn from_u16(value: u16) -> Option<Self> {
        let op = match value {
            0x00 => Self::AbsF,
            0x01 => Self::AbsI,
            0x02 => Self::AddF,
            0x03 => Self::AddI,
            0x04 => Self::AndI,
            0x05 => Self::Assert,
            0x06 => Self::Assume,
            0x07 => Self::AtomicCAS,
            0x08 => Self::AtomicRMW,
            0x09 => Self::Bitcast,
            0x0A => Self::Break,
            0x0B => Self::Broadcast,
            0x0C => Self::Cat,
            0x0D => Self::Ceil,
            0x0E => Self::CmpF,
            0x0F => Self::CmpI,
            0x10 => Self::Constant,
            0x11 => Self::Continue,
            0x12 => Self::Cos,
            0x13 => Self::CosH,
            0x14 => Self::DivF,
            0x15 => Self::DivI,
            0x16 => Self::Entry,
            0x17 => Self::Exp,
            0x18 => Self::Exp2,
            0x25 => Self::ExtI,
            0x26 => Self::Extract,
            0x27 => Self::Floor,
            0x28 => Self::Fma,
            0x29 => Self::For,
            0x2A => Self::FToF,
            0x2B => Self::FToI,
            0x2C => Self::GetGlobal,
            0x2D => Self::GetIndexSpaceShape,
            0x2E => Self::GetNumTileBlocks,
            0x2F => Self::GetTensorShape,
            0x30 => Self::GetTileBlockId,
            0x31 => Self::Global,
            0x32 => Self::If,
            0x33 => Self::IntToPtr,
            0x3A => Self::Iota,
            0x3B => Self::IToF,
            0x3C => Self::JoinTokens,
            0x3D => Self::LoadPtrTko,
            0x3E => Self::LoadViewTko,
            0x3F => Self::Log,
            0x40 => Self::Log2,
            0x41 => Self::Loop,
            0x42 => Self::MakePartitionView,
            0x43 => Self::MakeTensorView,
            0x44 => Self::MakeToken,
            0x45 => Self::MaxF,
            0x46 => Self::MaxI,
            0x47 => Self::MinF,
            0x48 => Self::MinI,
            0x49 => Self::MmaF,
            0x4A => Self::MmaI,
            0x4B => Self::Module,
            0x4C => Self::MulF,
            0x4D => Self::MulhiI,
            0x4E => Self::MulI,
            0x4F => Self::NegF,
            0x50 => Self::NegI,
            0x51 => Self::Offset,
            0x52 => Self::OrI,
            0x53 => Self::Permute,
            0x54 => Self::Pow,
            0x55 => Self::Print,
            0x56 => Self::PtrToInt,
            0x57 => Self::PtrToPtr,
            0x58 => Self::Reduce,
            0x59 => Self::RemF,
            0x5A => Self::RemI,
            0x5B => Self::Reshape,
            0x5C => Self::Return,
            0x5D => Self::Rsqrt,
            0x5E => Self::Scan,
            0x5F => Self::Select,
            0x60 => Self::ShLI,
            0x61 => Self::ShRI,
            0x62 => Self::Sin,
            0x63 => Self::SinH,
            0x64 => Self::Sqrt,
            0x65 => Self::StorePtrTko,
            0x66 => Self::StoreViewTko,
            0x67 => Self::SubF,
            0x68 => Self::SubI,
            0x69 => Self::Tan,
            0x6A => Self::TanH,
            0x6B => Self::TruncI,
            0x6C => Self::XOrI,
            0x6D => Self::Yield,
            0x6E => Self::Atan2,
            0x6F => Self::Pack,
            0x70 => Self::Unpack,
            0x71 => Self::Alloca,
            0x72 => Self::MmaFScaled,
            0x73 => Self::MakeGatherScatterView,
            0x74 => Self::MakeStridedView,
            0x75 => Self::AtomicRedViewTko,
            0x76 => Self::Insert,
            0x77 => Self::GdcLaunchDependentsTko,
            0x78 => Self::GdcWaitTko,
            0x79 => Self::FPowI,
            0x7A => Self::MemoryFenceAliasTko,
            0x7B => Self::MultimemLoadReduceViewTko,
            0x7C => Self::MultimemReduceViewTko,
            0x7D => Self::MultimemStoreViewTko,
            0x4002 => Self::PreviewCall,
            _ => return None,
        };
        Some(op)
    }

    /// Return the fixed result count if this op uses a fixed count in the
    /// bytecode format, or `None` if the op writes a varint result count.
    ///
    /// Derived from the generated Bytecode.inc: ops that call
    /// `writeVarInt(op->getNumResults())` return None; all others return
    /// the fixed count from Ops.td.
    pub fn fixed_result_count(&self) -> Option<usize> {
        use Opcode::*;
        match self {
            // Ops that write varint numResults (from Bytecode.inc audit):
            Break
            | Continue
            | Extract
            | Insert
            | For
            | GetIndexSpaceShape
            | GetTensorShape
            | If
            | JoinTokens
            | LoadViewTko
            | Loop
            | MakeTensorView
            | Print
            | Reduce
            | Return
            | Scan
            | StoreViewTko
            | Yield
            | MultimemLoadReduceViewTko
            | MultimemReduceViewTko
            | MultimemStoreViewTko
            | PreviewCall => None,

            // Fixed-count ops (from Ops.td):
            // 0 results
            Assert | Global | Module => Some(0),
            // 1 result
            AbsF
            | AbsI
            | AddF
            | AddI
            | AndI
            | Assume
            | Atan2
            | Bitcast
            | Broadcast
            | Cat
            | Ceil
            | CmpF
            | CmpI
            | Constant
            | Cos
            | CosH
            | DivF
            | DivI
            | Exp
            | Exp2
            | ExtI
            | Floor
            | FPowI
            | GdcLaunchDependentsTko
            | GdcWaitTko
            | Fma
            | FToF
            | FToI
            | GetGlobal
            | IntToPtr
            | Iota
            | IToF
            | Log
            | Log2
            | MakeGatherScatterView
            | MakePartitionView
            | MakeStridedView
            | MakeToken
            | MemoryFenceAliasTko
            | MaxF
            | MaxI
            | MinF
            | MinI
            | MmaF
            | MmaFScaled
            | MmaI
            | MulF
            | MulhiI
            | MulI
            | NegF
            | NegI
            | Offset
            | OrI
            | Pack
            | Permute
            | Pow
            | PtrToInt
            | PtrToPtr
            | Reshape
            | RemF
            | RemI
            | Rsqrt
            | Select
            | ShLI
            | ShRI
            | Sin
            | SinH
            | Sqrt
            | SubF
            | SubI
            | Tan
            | TanH
            | TruncI
            | Unpack
            | XOrI => Some(1),
            // 2 results
            AtomicCAS | AtomicRMW | LoadPtrTko => Some(2),
            // 1 result (token)
            AtomicRedViewTko | StorePtrTko => Some(1),
            // 3 results
            GetTileBlockId | GetNumTileBlocks => Some(3),
            Alloca => Some(1),
            // Entry has 0 results but is function-like (handled in func section, not here)
            Entry => Some(0),
        }
    }
}
