// The short causal convolution block: one channel's k-tap causal conv over its
// newest inputs — the decode conv of LFM2's conv layers and of Gated DeltaNet's
// prologue (`docs/fusion.md` §12.2 row 1, `docs/sequence-state.md` §10.2).
//
// A kernel composes three parts and keeps its own thread mapping:
//
//   taps      where the window's k inputs are read from. Tap 0 is the oldest,
//             tap k−1 the newest input.
//               short_conv_shift_taps — a `[k]` window whose slots 1..k−1 are
//                 the history, and the new input;
//               short_conv_seq_taps  — position `t` of a multi-position pass:
//                 the old window before position 0, the pass's rows after it.
//   fold      a wide accumulator over the taps against the weight row, oldest
//             first — the order every recorded digest of these kernels has.
//               short_conv_fold_into — also writes the window after the step
//                 (out of place: the state layout the shift arms carry);
//               short_conv_fold      — reads only;
//               short_conv_ring_fold — the column ring (§10.2a): the input
//                 written in place at its phase slot, then every slot folded
//                 in slot order against the phase's row of the permuted table,
//                 a different reduction order (§10.2g).
//   epilogue  the kernel's: LFM2 gates by `C`, the GDN prologue applies SiLU
//             and the head norms.
//
// Every function here is one channel's work and uses no threadgroup memory,
// so it composes with any thread mapping. The loop shapes are the kernels'
// own and are part of the result: under fast math the compiler picks a
// summation tree per shape, so a restructured loop is a different kernel.
//
// A text fragment with no `#include`: `source.rs` splices it after the
// consumer's header, once per translation unit.

template <typename T>
struct short_conv_shift_taps {
  const device T *win; // this channel's `[k]` window
  T x;                 // the new input

  METAL_FUNC T history(uint j, uint) const { return win[j + 1]; }
  METAL_FUNC T newest() const { return x; }
};

template <typename T>
struct short_conv_seq_taps {
  const device T *win;  // this channel's `[k]` window before position 0
  const device T *rows; // this channel's input at position 0
  uint stride;          // elements between two positions' inputs
  uint t;               // the position

  // The input `j` taps before the newest, or the window's slot for it.
  METAL_FUNC T at(uint j, uint k) const {
    const int tp = (int)t - (int)(k - 1) + (int)j;
    return tp < 0 ? win[(uint)((int)k + tp)] : rows[(size_t)tp * stride];
  }
  METAL_FUNC T history(uint j, uint k) const { return at(j, k); }
  METAL_FUNC T tap(uint j, uint k) const { return at(j, k); }
  METAL_FUNC T newest() const { return rows[(size_t)t * stride]; }
};

// The fold of `taps` against `w`, oldest first, writing the window after the
// step to `next` (never the window being read). One loop, the input selected
// in its last trip: split into a history loop and a trailing input term, the
// emulated-bfloat instantiation rounds its gated output differently.
template <typename A, typename Taps, typename T, typename W>
METAL_FUNC A short_conv_fold_into(thread const Taps &taps, W w, device T *next, uint k) {
  A acc = static_cast<A>(0);
#pragma clang loop unroll(full)
  for (uint j = 0; j < k; ++j) {
    T v;
    if (j + 1 < k) {
      v = taps.history(j, k);
    } else {
      v = taps.newest();
    }
    next[j] = v;
    acc += static_cast<A>(v) * static_cast<A>(w[j]);
  }
  return acc;
}

// The fold of `taps` against `w`, oldest first, reading only.
template <typename A, typename T, typename W>
METAL_FUNC A short_conv_fold(thread const short_conv_seq_taps<T> &taps, W w, uint k) {
  A acc = static_cast<A>(0);
  for (uint j = 0; j < k; ++j) {
    acc += static_cast<A>(w[j]) * static_cast<A>(taps.tap(j, k));
  }
  return acc;
}

// The column ring (§10.2a): `ring` is this channel's `[width]` row, written in
// place — `x` into slot `phase` — then folded over every slot in slot order
// against `w_phase`, the permuted table's row for `phase` (zero off the live
// window). The pragma pins the serial order: under fast math a branch-free
// loop may otherwise be split into interleaved partial sums.
template <typename A, typename T>
METAL_FUNC A short_conv_ring_fold(
    device T *ring,
    const device T *w_phase,
    size_t width,
    size_t phase,
    T x) {
  ring[phase] = x;
  A acc = static_cast<A>(0);
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
  for (size_t s = 0; s < width; ++s) {
    acc += static_cast<A>(ring[s]) * static_cast<A>(w_phase[s]);
  }
  return acc;
}
