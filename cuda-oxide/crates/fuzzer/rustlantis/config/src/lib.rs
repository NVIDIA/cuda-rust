use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub fn load(path: impl AsRef<Path>) -> Config {
    let config = std::fs::read_to_string(path).unwrap();
    toml::from_str(&config).unwrap()
}

#[derive(Deserialize, Clone)]
pub struct Config {
    #[serde(flatten)]
    pub generation: GenerationConfig,
    #[serde(flatten)]
    pub ty: TyConfig,
    pub backends: HashMap<String, BackendConfig>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "kebab-case")]
#[serde(tag = "type")]
pub enum BackendConfig {
    Miri {
        toolchain: String,
        flags: Vec<String>,
    },
    MiriRepo {
        path: String,
        flags: Vec<String>,
    },
    #[serde(rename = "llvm")]
    LLVM {
        toolchain: String,
        flags: Vec<String>,
    },
    Cranelift {
        toolchain: String,
        flags: Vec<String>,
    },
    CraneliftRepo {
        path: String,
        flags: Vec<String>,
    },
    CraneliftBinary {
        path: String,
        flags: Vec<String>,
    },
    #[serde(rename = "gcc")]
    GCC {
        path: String,
        flags: Vec<String>,
    },
}

#[derive(Deserialize, Clone)]
pub struct GenerationConfig {
    /// Max. number of statements & declarations in a bb
    #[serde(default = "bb_max_len")]
    pub bb_max_len: usize,

    /// Max. number of switch targets in a SwitchInt terminator
    #[serde(default = "max_switch_targets")]
    pub max_switch_targets: usize,

    /// Max. number of BB in a function if RET is init (a Return must be generated)
    #[serde(default = "max_bb_count")]
    pub max_bb_count: usize,

    /// Max. number of BB in a function before giving up this function
    #[serde(default = "max_bb_count_hard")]
    pub max_bb_count_hard: usize,

    /// Max. number of functions in the program Call generator stops being a possible candidate
    #[serde(default = "max_fn_count")]
    pub max_fn_count: usize,

    /// Max. number of arguments a function can have
    #[serde(default = "max_args_count")]
    pub max_args_count: usize,

    /// Expected proportion of variables to be dumped
    #[serde(default = "var_dump_chance")]
    pub var_dump_chance: f32,
}

fn bb_max_len() -> usize {
    32
}

fn max_switch_targets() -> usize {
    8
}

fn max_bb_count() -> usize {
    50
}

fn max_bb_count_hard() -> usize {
    100
}

fn max_fn_count() -> usize {
    20
}

fn max_args_count() -> usize {
    12
}

fn var_dump_chance() -> f32 {
    0.5
}

#[derive(Deserialize, Clone)]
pub struct TyConfig {
    /// Max. arity of tuple
    #[serde(default = "tuple_max_len")]
    pub tuple_max_len: usize,

    /// Max. len of array
    #[serde(default = "array_max_len")]
    pub array_max_len: usize,

    /// Max. number of fields in a struct or enum variant
    #[serde(default = "struct_max_fields")]
    pub struct_max_fields: usize,

    /// Max. number of variants in an enum
    #[serde(default = "adt_max_variants")]
    pub adt_max_variants: usize,

    /// Number of composite structural types
    #[serde(default = "composite_count")]
    pub composite_count: usize,

    /// Number of ADTs
    #[serde(default = "adt_count")]
    pub adt_count: usize,
}

fn tuple_max_len() -> usize {
    4
}
fn array_max_len() -> usize {
    8
}

fn struct_max_fields() -> usize {
    8
}

fn adt_max_variants() -> usize {
    4
}

fn composite_count() -> usize {
    64
}

fn adt_count() -> usize {
    8
}

impl Default for TyConfig {
    fn default() -> Self {
        Self {
            tuple_max_len: 4,
            array_max_len: 8,
            struct_max_fields: 8,
            adt_max_variants: 4,
            composite_count: 64,
            adt_count: 8,
        }
    }
}

#[test]
fn example_parses() {
    let _ = load("../config.toml.example");
}
