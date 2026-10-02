fn main() -> std::process::ExitCode {
    eprintln!("expected stderr");
    std::process::ExitCode::from(1)
}
