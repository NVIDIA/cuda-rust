use std::{
    collections::HashMap,
    env,
    ffi::{OsStr, OsString},
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    process::{self, Command, ExitStatus, Stdio},
};

use log::debug;

use crate::Source;
use config::BackendConfig;
use config::Config;

trait ClearEnv {
    fn clear_env(&mut self, preserve: &[&str]) -> &mut Command;
}

impl ClearEnv for Command {
    fn clear_env(&mut self, preserve: &[&str]) -> &mut Command {
        self.env_clear();
        for env in preserve {
            if let Ok(existing) = env::var(env) {
                self.env(env, existing);
            }
        }
        self
    }
}

#[derive(serde::Deserialize)]
struct RustcDiagnostic {
    #[serde(rename = "$message_type")]
    message_type: String,
    message: String,
    level: String,
    rendered: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ProcessOutput {
    pub status: ExitStatus,
    /// The data that the process wrote to stdout.
    pub stdout: OsString,
    /// The data that the process wrote to stderr.
    pub stderr: OsString,
}
impl Hash for ProcessOutput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.status.code().hash(state);
        self.stdout.hash(state);
        self.stderr.hash(state);
    }
}

impl From<process::Output> for ProcessOutput {
    fn from(value: process::Output) -> Self {
        let stdout: OsString;
        let stderr: OsString;
        #[cfg(unix)]
        {
            use std::os::unix::prelude::OsStrExt;
            stdout = OsStr::from_bytes(&value.stdout).to_owned();
            stderr = OsStr::from_bytes(&value.stderr).to_owned();
        }
        #[cfg(windows)]
        {
            use std::os::windows::prelude::OsStrExt;
            stdout = OsStr::from_wide(&value.stdout).to_owned();
            stderr = OsStr::from_wide(&value.stderr).to_owned();
        }
        Self {
            status: value.status,
            stdout,
            stderr,
        }
    }
}

pub fn from_config(config: Config) -> HashMap<String, Box<dyn Backend>> {
    let mut backends = HashMap::new();
    for (name, config) in config.backends {
        let backend: Box<dyn Backend> = match config {
            BackendConfig::Miri { toolchain, flags } => {
                Box::new(Miri::from_rustup(toolchain, flags).unwrap())
            }
            BackendConfig::MiriRepo { path, flags } => {
                Box::new(Miri::from_repo(path, flags).unwrap())
            }
            BackendConfig::LLVM { toolchain, flags } => Box::new(Llvm::new(toolchain, flags)),
            BackendConfig::Cranelift { toolchain, flags } => {
                Box::new(Cranelift::from_rustup(toolchain, flags))
            }
            BackendConfig::CraneliftRepo { path, flags } => {
                Box::new(Cranelift::from_repo(path, flags).unwrap())
            }
            BackendConfig::CraneliftBinary { path, flags } => {
                Box::new(Cranelift::from_binary(path, flags))
            }
            BackendConfig::GCC { path, flags } => {
                Box::new(Gcc::from_built_repo(path, flags).unwrap())
            }
        };
        backends.insert(name, backend);
    }
    backends
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct CompExecError(pub ProcessOutput);

pub type ExecResult = Result<ProcessOutput, CompExecError>;

#[derive(Debug)]
pub struct BackendInitError(pub String);

pub trait Backend: Send + Sync {
    fn compile(&self, _: &Source, _: &Path) -> ProcessOutput {
        unimplemented!("this backend does not support compilation")
    }

    fn execute(&self, source: &Source, target: &Path) -> ExecResult {
        debug!("Compiling {source}");
        let compile_out = self.compile(source, target);
        if !compile_out.status.success() {
            return Err(CompExecError(compile_out));
        }

        debug!("Executing compiled {source}");
        let exec_out = Command::new(target)
            .output()
            .expect("can execute target program and get output");
        Ok(exec_out.into())
    }
}

fn run_compile_command(mut command: Command, source: &Source) -> process::Output {
    let compiler = match source {
        Source::File(path) => {
            command.arg(path.canonicalize().expect("path is correct"));
            command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("can spawn compiler")
        }
        Source::Stdin(code) => {
            command.arg("-").stdin(Stdio::piped());
            let mut child = command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("can spawn compiler");
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(code.as_bytes())
                .unwrap();
            child
        }
    };

    compiler
        .wait_with_output()
        .expect("can execute rustc and get output")
}

struct Llvm {
    toolchain: String,
    flags: Vec<String>,
}

impl Llvm {
    fn new(toolchain: String, flags: Vec<String>) -> Self {
        Self { toolchain, flags }
    }
}

impl Backend for Llvm {
    fn compile(&self, source: &Source, target: &Path) -> ProcessOutput {
        let mut command = Command::new("rustc");

        command.arg(format!("+{}", self.toolchain));

        command
            .args(["-o", target.to_str().unwrap()])
            .args(["-C", "llvm-args=-protect-from-escaped-allocas=true"]) // https://github.com/rust-lang/rust/issues/112213
            .args(self.flags.clone());

        run_compile_command(command, source).into()
    }
}

enum BackendSource {
    Path(PathBuf),
    Rustup(String),
}

struct Miri {
    miri: BackendSource,
    sysroot: PathBuf,
    flags: Vec<String>,
}

impl Miri {
    fn find_sysroot(miri_source: &BackendSource) -> Result<PathBuf, BackendInitError> {
        let mut command = match miri_source {
            BackendSource::Path(source_dir) => {
                let mut cmd = Command::new(source_dir.join("target/release/cargo-miri"));
                cmd.current_dir(source_dir);
                cmd
            }
            BackendSource::Rustup(toolchain) => {
                let mut cmd = Command::new("rustup");
                cmd.args(["run", toolchain, "cargo-miri"]);
                cmd
            }
        };
        let output = command
            .arg("miri")
            .arg("setup")
            .arg("--print-sysroot")
            .clear_env(&["PATH", "DEVELOPER_DIR"])
            .output()
            .expect("can run cargo-miri setup --print-sysroot");
        if !output.status.success() {
            return Err(BackendInitError(format!(
                "failed to find sysroot: {output:?}"
            )));
        }
        let sysroot;
        #[cfg(unix)]
        {
            use std::os::unix::prelude::OsStrExt;
            sysroot = OsStr::from_bytes(output.stdout.trim_ascii_end()).to_owned();
        }
        #[cfg(windows)]
        {
            use std::os::windows::prelude::OsStrExt;
            sysroot = OsStr::from_wide(output.stdout.trim_ascii_end()).to_owned();
        }

        let sysroot = PathBuf::from(sysroot);

        debug!("Miri sysroot at {}", sysroot.to_string_lossy());
        if !Path::exists(&sysroot) {
            return Err(BackendInitError("sysroot does not exist".to_string()));
        }

        Ok(sysroot)
    }

    fn from_repo<P: AsRef<Path>>(
        miri_dir: P,
        flags: Vec<String>,
    ) -> Result<Self, BackendInitError> {
        let miri_dir = miri_dir.as_ref();

        // Detect if Miri already built
        if !Path::exists(&miri_dir.join("target/release/cargo-miri"))
            || !Path::exists(&miri_dir.join("target/release/miri"))
        {
            // Otherwise, build it ourselves
            debug!("Setting up miri toolchain");
            let output = Command::new(miri_dir.join("miri"))
                .arg("toolchain")
                .output()
                .expect("can run miri toolchain and get output");
            if !output.status.success() {
                return Err(BackendInitError(format!(
                    "failed to set up Miri toolchain: {output:?}"
                )));
            }

            debug!("Building Miri under {}", miri_dir.to_string_lossy());
            let output = Command::new(miri_dir.join("miri"))
                .arg("build")
                .arg("--release")
                .clear_env(&["PATH", "DEVELOPER_DIR"])
                .current_dir(miri_dir)
                .output()
                .expect("can run miri build and get output");
            if !output.status.success() {
                return Err(BackendInitError(format!(
                    "failed to build Miri: {output:?}"
                )));
            }
        } else {
            debug!("Detected built Miri under {}", miri_dir.to_string_lossy());
        }

        let sysroot = Self::find_sysroot(&BackendSource::Path(miri_dir.to_owned()))?;

        Ok(Self {
            miri: BackendSource::Path(miri_dir.join("target/release/miri")),
            sysroot,
            flags,
        })
    }

    fn from_rustup(toolchain: String, flags: Vec<String>) -> Result<Self, BackendInitError> {
        let sysroot = Self::find_sysroot(&BackendSource::Rustup(toolchain.to_owned()))?;

        Ok(Self {
            miri: BackendSource::Rustup(toolchain),
            sysroot,
            flags,
        })
    }
}

fn parse_miri_stderr(stderr: Vec<u8>) -> (Vec<u8>, bool) {
    let mut rendered_stderr = Vec::with_capacity(stderr.len());
    let mut has_error = false;

    for line in stderr.split_inclusive(|byte| *byte == b'\n') {
        let json = line.strip_suffix(b"\n").unwrap_or(line);

        match serde_json::from_slice::<RustcDiagnostic>(json) {
            Ok(diagnostic) if diagnostic.message_type == "diagnostic" => {
                has_error |= diagnostic.level == "error" || diagnostic.level.starts_with("error:");

                if let Some(rendered) = diagnostic.rendered {
                    rendered_stderr.extend_from_slice(rendered.as_bytes());

                    if !rendered.ends_with('\n') {
                        rendered_stderr.push(b'\n');
                    }
                } else {
                    rendered_stderr.extend_from_slice(diagnostic.message.as_bytes());
                    rendered_stderr.push(b'\n');
                }
            }
            _ => {
                // Preserve stderr produced by the interpreted program.
                rendered_stderr.extend_from_slice(line);
            }
        }
    }

    (rendered_stderr, has_error)
}

impl Backend for Miri {
    fn execute(&self, source: &Source, _: &Path) -> ExecResult {
        debug!("Executing with Miri {source}");
        let mut command = match &self.miri {
            BackendSource::Path(binary) => Command::new(binary),
            BackendSource::Rustup(toolchain) => {
                let mut cmd = Command::new("rustup");
                cmd.args(["run", toolchain, "miri"]);
                cmd
            }
        };
        command.args(self.flags.clone());

        command
            .clear_env(&["PATH", "DEVELOPER_DIR"])
            .args([OsStr::new("--sysroot"), self.sysroot.as_os_str()])
            .arg("--error-format=json");

        let mut miri_out = run_compile_command(command, source);
        let (stderr, miri_failed) = parse_miri_stderr(miri_out.stderr);

        miri_out.stderr = stderr;

        if miri_failed || miri_out.status.code().is_none() {
            Err(CompExecError(miri_out.into()))
        } else {
            Ok(miri_out.into())
        }
    }
}

struct Cranelift {
    clif: BackendSource,
    flags: Vec<String>,
}

impl Cranelift {
    fn from_repo<P: AsRef<Path>>(
        clif_dir: P,
        flags: Vec<String>,
    ) -> Result<Self, BackendInitError> {
        let clif_dir = clif_dir.as_ref();

        if !Path::exists(&clif_dir.join("dist/rustc-clif")) {
            debug!("Setting up cranelift under {}", clif_dir.to_string_lossy());
            let output = Command::new(clif_dir.join("y.rs"))
                .arg("prepare")
                .clear_env(&["PATH", "DEVELOPER_DIR"])
                .current_dir(clif_dir)
                .output()
                .expect("can run y.rs prepare and get output");
            if !output.status.success() {
                return Err(BackendInitError(format!(
                    "failed to prepare Cranelift: {output:?}"
                )));
            }

            let output = Command::new(clif_dir.join("y.rs"))
                .arg("build")
                .clear_env(&["PATH", "DEVELOPER_DIR"])
                .current_dir(clif_dir)
                .output()
                .expect("can run y.rs build and get output");
            if !output.status.success() {
                return Err(BackendInitError(format!(
                    "failed to build Cranelift: {output:?}"
                )));
            }
        } else {
            debug!("Found built Cranelift under {}", clif_dir.to_string_lossy());
        }

        Ok(Cranelift {
            clif: BackendSource::Path(clif_dir.join("dist/rustc-clif")),
            flags,
        })
    }

    fn from_binary<P: AsRef<Path>>(binary_path: P, flags: Vec<String>) -> Self {
        Self {
            clif: BackendSource::Path(binary_path.as_ref().to_owned()),
            flags,
        }
    }

    fn from_rustup(toolchain: String, flags: Vec<String>) -> Self {
        Self {
            clif: BackendSource::Rustup(toolchain),
            flags,
        }
    }
}

impl Backend for Cranelift {
    fn compile(&self, source: &Source, target: &Path) -> ProcessOutput {
        let mut command = match &self.clif {
            BackendSource::Path(binary) => Command::new(binary),
            BackendSource::Rustup(toolchain) => {
                let mut cmd = Command::new("rustc");
                cmd.arg(format!("+{toolchain}"));
                cmd.args(["-Z", "codegen-backend=cranelift"]);
                cmd
            }
        };
        command
            .args(["-o", target.to_str().unwrap()])
            .args(self.flags.clone());
        run_compile_command(command, source).into()
    }
}

struct Gcc {
    library: PathBuf,
    sysroot: PathBuf,
    repo: PathBuf,
    flags: Vec<String>,
}

impl Gcc {
    fn from_built_repo<P: AsRef<Path>>(
        cg_gcc: P,
        flags: Vec<String>,
    ) -> Result<Self, BackendInitError> {
        let Ok(cg_gcc) = cg_gcc.as_ref().to_owned().canonicalize() else {
            return Err(BackendInitError(
                "cannot rustc_codegen_gcc repo".to_string(),
            ));
        };

        let Ok(library) = cg_gcc
            .join("target/release/librustc_codegen_gcc.so")
            .canonicalize()
        else {
            return Err(BackendInitError(
                "cannot find librustc_codegen_gcc.so".to_string(),
            ));
        };
        let Ok(sysroot) = cg_gcc.join("build_sysroot/sysroot").canonicalize() else {
            return Err(BackendInitError("cannot find sysroot".to_string()));
        };

        Ok(Self {
            library,
            sysroot,
            repo: cg_gcc,
            flags,
        })
    }
}

impl Backend for Gcc {
    fn compile(&self, source: &Source, target: &Path) -> ProcessOutput {
        let mut command = Command::new("rustc");
        command
            .clear_env(&["PATH", "DEVELOPER_DIR", "LD_LIBRARY_PATH"])
            .current_dir(&self.repo)
            .args([
                "-Z",
                &format!("codegen-backend={}", self.library.to_str().unwrap()),
            ])
            .arg("--sysroot")
            .arg(&self.sysroot)
            .args(["-o", target.to_str().unwrap()])
            .args(self.flags.clone());
        run_compile_command(command, source).into()
    }
}
