//! `lloom-serve` — `lloom serve` under the name the pipette runner spawns
//! (`Runtime::LloomServeMacos`, issue #702 §4). See `lloom_engine::app`.
fn main() -> anyhow::Result<()> {
    lloom_engine::app::main_serve_compat(std::env::args().collect())
}
