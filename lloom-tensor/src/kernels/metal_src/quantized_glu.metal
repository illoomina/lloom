// The routed SwiGLU on the gate|up pair's store (`kernels.fusion.routed_glu`,
// `docs/expert-streaming.md` §20.15f).
//
// `kernel_mul_mv_id2`'s plane decode with both halves of a row tile in one
// threadgroup: the gate core writes the tile into `dst[0]` and the up core
// into `dst[1]`, each exactly as the pair stores them, and the threadgroup
// then overwrites its tile of `dst[0]` with `silu_times(g, u)` read back from
// those stores -- `silu_mul`'s element on the values `silu_mul` would read, so
// the launch is the pair then `silu_mul` bit for bit. `dst[1]` keeps `u`.
//
// `tile` is the rows one threadgroup of the single launch writes (the host's
// `mv_threadgroup` align): threadgroup `x` writes rows `[x·tile, (x+1)·tile)`
// below `ne0`, and reads back only those. The barrier between the cores keeps
// a codebook staged into `shared_values` by the first from being restaged
// under a reader; the device barrier orders the cores' stores before the
// read-back by any thread of the group.
template<typename D, typename mmv_store<D>::fn_t impl_fn>
kernel void kernel_mul_mv_id2_glu(
        device const    char * src0a,
        device const    char * src0b,
        device const    char * src1,
        device             D * dst,
        device const    char * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant     int64_t & tile,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]],
        uint3                  ntg[[threads_per_threadgroup]],
        EOS_GATE_ARG) {
    EOS_GATE_CHECK
    const int z = tgpig.z;
    const int iid1 = z/nei0;
    const int idx = z%nei0;
    const int64_t r0 = (int64_t)tgpig.x * tile;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*nbi1))[idx];

    const int64_t i11 = idx % ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src1_cur = src1 + i11*nb11 + i12*nb12;
    device          D * dst_g    = dst + i1*ne0 + i2*ne1*ne0;
    device          D * dst_u    = dst_g + nei1*ne1*ne0;

    impl_fn(src0a + i02*nb02, src1_cur, dst_g, ne00, ne01, 1, nb00, nb01, nb02, ne10, 1, 1, 1,
            nb10, nb11, nb12, ne0, 1, nb1, 1, 1, shared_values, tgpig, tiitg, tiisg, sgitg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    impl_fn(src0b + i02*nb02, src1_cur, dst_u, ne00, ne01, 1, nb00, nb01, nb02, ne10, 1, 1, 1,
            nb10, nb11, nb12, ne0, 1, nb1, 1, 1, shared_values, tgpig, tiitg, tiisg, sgitg);
    threadgroup_barrier(mem_flags::mem_device);

    const int64_t r1 = r0 + tile < ne0 ? r0 + tile : ne0;
    const uint nth = ntg.x * ntg.y * ntg.z;
    for (int64_t r = r0 + tiitg; r < r1; r += nth) {
        dst_g[r] = silu_times(dst_g[r], dst_u[r]);
    }
}
