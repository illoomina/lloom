#include <metal_stdlib>

using namespace metal;

// Kernels the Qwen4-exp blocks dispatch (`lloom::models::qwen4exp`): the
// pieces of the decode and prefill paths that were composed from tens of
// elementwise ops, each written as ONE launch. Every kernel here has a
// scalar reference in the model crate it is tested against.

// `(I + A)⁻¹` for a strictly-lower-triangular `A [batch, n, n]` (row-major),
// by forward substitution — `T[i, :] = e_i − Σ_{m<i} A[i, m] · T[m, :]`, each
// row from the rows before it (the numerically stable form; the nilpotent
// product overflows on real keys, see `deltanet::unit_lower_inverse`).
//
// One threadgroup per matrix; thread `j` owns column `j` (and every column
// `j + k·width`), rows run in order with a device-memory barrier between
// them, so row `i` reads the rows `< i` the group has already stored. `n`
// is the chunk (64): 64 rows × 64 threads, ~2k FMAs a thread. The composed
// path this replaces was an `n`-iteration host loop of `narrow`/`cat`/
// `matmul`, ~2k `copy2d` launches per matrix batch (the prefill's storm).
kernel void unit_lower_inverse_f32(
    const device float* a [[buffer(0)]],
    device float* t [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]],
    uint j [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  const device float* am = a + (size_t)b * n * n;
  device float* tm = t + (size_t)b * n * n;
  for (uint i = 0; i < n; i++) {
    for (uint jj = j; jj < n; jj += width) {
      float acc = (i == jj) ? 1.0f : 0.0f;
      for (uint m = 0; m < i; m++) {
        acc -= am[i * n + m] * tm[m * n + jj];
      }
      tm[i * n + jj] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
