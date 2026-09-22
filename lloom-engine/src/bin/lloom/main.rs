//! `lloom serve | chat` — see `lloom_engine::app`.
fn main() -> anyhow::Result<()> {
    lloom_engine::app::main(std::env::args().collect())
}
