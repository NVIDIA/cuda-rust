#![feature(iter_intersperse)]

use std::{
    io::{self, Read},
    path::PathBuf,
    process::ExitCode,
    str::FromStr,
};

use clap::{Arg, Command};
use difftest::backends;
use difftest::{Source, run_diff_test};
use log::{debug, error, info};

fn main() -> ExitCode {
    env_logger::init();

    let matches = Command::new("difftest")
        .arg(Arg::new("file").required(true))
        .get_matches();
    let source = matches.get_one::<String>("file").expect("required");

    let config_path = std::env::var("RUSTLANTIS_CONFIG").unwrap_or("config.toml".to_string());
    let config = config::load(config_path);
    let backends = backends::from_config(config);

    let source = if source == "-" {
        let mut code = String::new();
        io::stdin()
            .read_to_string(&mut code)
            .expect("can read source code from stdin");
        Source::Stdin(code)
    } else {
        Source::File(PathBuf::from_str(source).expect("is valid path"))
    };

    info!(
        "Difftesting {} with {}",
        source,
        backends
            .keys()
            .map(String::as_str)
            .intersperse(", ")
            .collect::<String>()
    );

    let results = run_diff_test(&source, backends);
    if results.all_same() && results.all_success() {
        info!("{} is all the same", source);
        debug!("{}", results);
        ExitCode::SUCCESS
    } else {
        let results = results.to_string();
        error!("{} didn't pass:\n{results}", source);
        ExitCode::FAILURE
    }
}
