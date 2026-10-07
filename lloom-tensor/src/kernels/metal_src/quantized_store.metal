// The quantized matvec's store as a block (`docs/optimizer-design.md` §10k).
//
// Every `kernel_mul_mv_*_impl` core writes each output element once, as
// `dst[i] = (STORE)(v)` (or an implicit conversion from the f32 sum), so a
// STORE type whose conversion or whose assignment carries the block rides the
// store with no core edited. Each is one float, the layout of the f32 store
// the host binds.
//
// `MvStoreSilu` converts `v` to `silu(v)` — the body of `utils.metal`'s
// `silu<T>` at float, the spelling the SwiGLU's `silu_mul_f32` computes.
//
// `MvStoreMulInto` multiplies `v` into the value the destination already
// holds: bound to the SwiGLU gate's own buffer, where the gate's store (or its
// SiLU launch) wrote `silu(g)`, the up projection lands as `silu(g) · u` — the
// product in `silu_mul`'s order. The element is read and written by the one
// thread that stores it; the gate must have no other reader.
struct MvStoreSilu {
  float v;
  MvStoreSilu(float x) : v(x / (1 + exp(-x))) {}
};

struct MvStoreMulInto {
  float v;
  MvStoreMulInto(float x) : v(x) {}
  void operator=(thread const MvStoreMulInto& o) device { v = v * o.v; }
};
