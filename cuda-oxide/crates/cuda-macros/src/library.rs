// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Register native inputs during expansion, before rustc writes dep-info.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MAX_JSON: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    version: u32,
    provider: String,
    executable: PathBuf,
    operation: String,
    algorithm: String,
}

#[derive(Serialize)]
struct InputRequest {
    version: u32,
    cuda_root: PathBuf,
    target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputResponse {
    version: u32,
    paths: Vec<PathBuf>,
    environment: Vec<String>,
}

fn environment(name: &str) -> Result<Option<String>, String> {
    match proc_macro::tracked::env_var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("cannot read {name}: {error}")),
    }
}

fn registration(value: &str) -> Result<Registration, String> {
    if value.len() > MAX_JSON {
        return Err("native library registration exceeds 1 MiB".into());
    }
    let registration: Registration = serde_json::from_str(value)
        .map_err(|error| format!("invalid native library registration: {error}"))?;
    if registration.version != 1
        || registration.provider.is_empty()
        || registration.operation.is_empty()
        || registration.algorithm.is_empty()
        || !registration.executable.is_absolute()
    {
        return Err("native library registration requires version 1, nonempty names and an absolute executable path".into());
    }
    Ok(registration)
}

fn response(bytes: &[u8]) -> Result<InputResponse, String> {
    if bytes.len() > MAX_JSON {
        return Err("native library input response exceeds 1 MiB".into());
    }
    let response: InputResponse = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid native library input response: {error}"))?;
    if response.version != 1 {
        return Err("native library input response requires version 1".into());
    }
    if response.paths.iter().any(|path| !path.is_absolute()) {
        return Err("native library input paths must be absolute".into());
    }
    if response
        .environment
        .iter()
        .any(|name| name.is_empty() || name.as_bytes().iter().any(|byte| matches!(byte, b'=' | 0)))
    {
        return Err(
            "native library input response has an invalid environment variable name".into(),
        );
    }
    Ok(response)
}

fn query(executable: &Path, request: &InputRequest) -> Result<InputResponse, String> {
    let bytes = serde_json::to_vec(request).map_err(|error| error.to_string())?;
    if bytes.len() > MAX_JSON {
        return Err("native library input request exceeds 1 MiB".into());
    }
    let mut child = Command::new(executable)
        .arg("--oxide-library-inputs-v1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| {
            format!(
                "cannot query native library {}: {error}",
                executable.display()
            )
        })?;
    let read = (|| {
        child.stdin.take().unwrap().write_all(&bytes)?;
        let mut output = Vec::new();
        child
            .stdout
            .take()
            .unwrap()
            .take((MAX_JSON + 1) as u64)
            .read_to_end(&mut output)?;
        Ok::<_, std::io::Error>(output)
    })();
    let output = match read {
        Ok(output) if output.len() <= MAX_JSON => output,
        other => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(match other {
                Err(error) => format!("native library input query failed: {error}"),
                _ => "native library input response exceeds 1 MiB".into(),
            });
        }
    };
    let status = child.wait().map_err(|error| error.to_string())?;
    if !status.success() {
        return Err(format!("native library input query exited with {status}"));
    }
    response(&output)
}

fn track_tree(
    path: &Path,
    directories: &mut BTreeSet<PathBuf>,
    track: &mut impl FnMut(&Path),
) -> Result<(), String> {
    let metadata = path.metadata().map_err(|error| {
        format!(
            "cannot track native library input {}: {error}",
            path.display()
        )
    })?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(format!(
            "native library input is not a file or directory: {}",
            path.display()
        ));
    }
    track(path);
    let canonical = path.canonicalize().map_err(|error| error.to_string())?;
    if canonical != path {
        track(&canonical);
    }
    if metadata.is_dir() && directories.insert(canonical) {
        for entry in std::fs::read_dir(path).map_err(|error| error.to_string())? {
            track_tree(
                &entry.map_err(|error| error.to_string())?.path(),
                directories,
                track,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn expand(variable: &syn::LitStr) -> Result<proc_macro2::TokenStream, String> {
    crate::common::track_codegen_environment();
    let value = environment(&variable.value())?.ok_or_else(|| {
        format!(
            "native library registration variable {} is not set",
            variable.value()
        )
    })?;
    let registration = registration(&value)?;
    if !registration.executable.is_file() {
        return Err(format!(
            "native library provider is not a file: {}",
            registration.executable.display()
        ));
    }
    let mut directories = BTreeSet::new();
    let mut track = |path: &Path| proc_macro::tracked::path(path);
    track_tree(&registration.executable, &mut directories, &mut track)?;
    let cuda_home = environment("CUDA_HOME")?;
    let cuda_path = environment("CUDA_PATH")?;
    let target = environment("CUDA_OXIDE_TARGET")?;
    if environment(reserved_oxide_symbols::MATERIALIZE_CUBIN_ENV)?
        .is_some_and(|value| matches!(value.as_str(), "1" | "true"))
    {
        let cuda_root = cuda_home
            .or(cuda_path)
            .ok_or("native library inputs require CUDA_HOME or CUDA_PATH")?;
        let cuda_root = std::path::absolute(cuda_root)
            .map_err(|error| format!("cannot resolve CUDA toolkit path: {error}"))?;
        let target = target.ok_or("native library inputs require CUDA_OXIDE_TARGET")?;
        let target = target
            .strip_prefix("sm_")
            .map(|value| format!("compute_{value}"))
            .unwrap_or(target);
        let inputs = query(
            &registration.executable,
            &InputRequest {
                version: 1,
                cuda_root,
                target,
            },
        )?;
        for name in inputs.environment {
            environment(&name)?;
        }
        for path in inputs.paths {
            track_tree(&path, &mut directories, &mut track)?;
        }
    }
    let literal = syn::LitStr::new(&value, variable.span());
    Ok(quote::quote!(#literal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_response_rejects_invalid_contracts() {
        for value in [
            r#"{"version":2,"paths":[],"environment":[]}"#,
            r#"{"version":1,"paths":["relative"],"environment":[]}"#,
            r#"{"version":1,"paths":[],"environment":["A=B"]}"#,
            r#"{"version":1,"paths":[],"environment":[],"extra":1}"#,
            r#"{"version":1,"paths":[],"environment":[]} trailing"#,
        ] {
            assert!(response(value.as_bytes()).is_err(), "{value}");
        }
        assert!(response(&vec![b' '; MAX_JSON + 1]).is_err());
        assert!(response(br#"{"version":1,"paths":[],"environment":[]}"#).is_ok());
    }
}
