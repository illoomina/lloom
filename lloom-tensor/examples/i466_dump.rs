//! Dump the source text the crate ACTUALLY compiles, one file per `Source`.
//!
//! Since issue #458 the `.metal` files under `metal_src/` are FRAGMENTS: seven
//! of them carry no `#include <metal_stdlib>` and no helper definitions, because
//! `source.rs` splices an include prologue and `utils.metal` ahead of each with
//! `concat!`. Reading `metal_src/*.metal` off disk therefore measures a program
//! that is not what ships and does not compile -- run against the post-#458 tree
//! the #452 pairwise probe fails its own control on 7 of 19 files for exactly
//! this reason (issue #466).
//!
//! So a probe that wants to measure the shipping tree has to get its text from
//! the crate rather than from the directory. This writes each `Source`'s
//! composed text to `<outdir>/<name>.metal`, keeping the `metal_src` basenames
//! so the existing #458 probe (`probes/issue-458/dedup.m`), which is keyed on
//! those filenames, can be pointed straight at the output.
//!
//! `utils.metal` is deliberately NOT emitted: it is not a `Source`, it is
//! composed INTO the seven consumers, and it does not compile alone.
//!
//! Usage: cargo run --release --example i466_dump -- <outdir>

fn main() -> std::io::Result<()> {
    let out = std::env::args().nth(1).expect("usage: i466_dump <outdir>");
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir)?;

    // (filename under metal_src, composed text). The filenames are the ones the
    // #458 probe's kernel table already keys on.
    use lloom_tensor::kernels::source as s;
    let sources: &[(&str, &str)] = &[
        ("affine.metal", s::AFFINE),
        ("arena_alloc.metal", s::ARENA_ALLOC),
        ("binary.metal", s::BINARY),
        ("cast.metal", s::CAST),
        ("conv.metal", s::CONV),
        ("fill.metal", s::FILL),
        ("flash_decoding.metal", s::FLASH_DECODING),
        ("gemv.metal", s::GEMV),
        ("indexing.metal", s::INDEXING),
        ("mlx_gemm.metal", s::MLX_GEMM),
        ("mlx_sort.metal", s::MLX_SORT),
        ("quantized.metal", s::QUANTIZED),
        ("random.metal", s::RANDOM),
        ("reduce.metal", s::REDUCE),
        ("scaled_dot_product_attention.metal", s::SDPA),
        ("scratch.metal", s::SCRATCH),
        ("sort.metal", s::SORT),
        ("ternary.metal", s::TERNARY),
        ("unary.metal", s::UNARY),
    ];

    for (name, text) in sources {
        std::fs::write(dir.join(name), text)?;
    }
    println!(
        "wrote {} composed sources to {}",
        sources.len(),
        dir.display()
    );
    Ok(())
}
