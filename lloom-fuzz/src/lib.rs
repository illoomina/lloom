//! LibAFL harnesses over lloom's stateful fuzz models (lloom #706).
//!
//! One binary per target in `src/bin/`; the models live in the engine crates
//! behind their `fuzzing` feature (the proptests drive the same ones), the
//! decoders in [`decode`], the campaign glue in [`campaign`], the persistent
//! record in [`record`]. Two models are this crate's own: [`kv_pages_model`]
//! (the paged pool, never fuzzed before) and [`cb_liveness`] (command-buffer
//! liveness under retained and unretained command buffers, with the pool's
//! free poison as the detector). [`geometry`] is the shape campaign over a
//! tiny model's geometry (`docs/porting-models.md` §21.8).

pub mod campaign;
pub mod decode;
pub mod record;

mod cmp_stubs;

#[cfg(feature = "metal")]
pub mod cb_liveness;
#[cfg(feature = "metal")]
pub mod geometry;
#[cfg(feature = "metal")]
pub mod kv_pages_model;
