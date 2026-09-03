#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Packed parameter blocks.
//
// An ICB command cannot carry an inline constant -- `MTLIndirectComputeCommand`
// has no `setBytes` in any form (`DESIGN.md` §3.7c) -- so every scalar a kernel
// takes as `constant size_t &n` has to arrive in a buffer instead. These are the
// structs the `_packed` entry points read, mirrored by `#[repr(C)]` types in
// `kernels/params.rs` and checked against them by `conv_params_layout`.
//
// `size_t` is 8 bytes in MSL, so most of these are 8-aligned -- unlike
// `reduce.metal`'s `uint` structs, which are 4-aligned. `UpsampleBilinear2dParams`
// is the one that mixes widths, and it is the reason the layout check ships the
// numbers across the boundary rather than trusting either side: it holds three
// `bool` (1 byte each in MSL) and two `float` between two `size_t`, so every
// field after the first is at an offset the padding rule decides.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length.
struct Im2colParams {
    size_t dst_numel;
    size_t h_out;
    size_t w_out;
    size_t h_k;
    size_t w_k;
    size_t stride;
    size_t padding;
    size_t dilation;
};

struct Col2im1dParams {
    size_t dst_el;
    size_t l_out;
    size_t l_in;
    size_t c_out;
    size_t k_size;
    size_t stride;
};

struct Im2col1dParams {
    size_t dst_numel;
    size_t l_out;
    size_t l_k;
    size_t stride;
    size_t padding;
    size_t dilation;
};

struct UpsampleNearest2dParams {
    size_t w_out;
    size_t h_out;
    float w_scale;
    float h_scale;
};

// The one struct here that mixes widths. `bool` is 1 byte in MSL as it is in
// Rust, and a `float` after it pads to 4 -- both hazards issue #38 names, in one
// struct. Field order mirrors the classical argument list exactly, so that the
// capture (which appends in bind order) produces this layout without anyone
// restating it.
struct UpsampleBilinear2dParams {
    size_t w_out;
    size_t h_out;
    bool align_corners;
    bool has_scale_h;
    float scale_h_factor;
    bool has_scale_w;
    float scale_w_factor;
};

// `max_pool2d` and `avg_pool2d` bind the same four scalars, so they share a
// struct. They differ in their *accumulator*, which is a template parameter and
// not part of the binding -- see the instantiation list, where the integer
// `avg_pool2d` rows deliberately accumulate in their own type.
struct Pool2dParams {
    size_t w_k;
    size_t h_k;
    size_t w_stride;
    size_t h_stride;
};

struct ConvTranspose1dParams {
    size_t l_out;
    size_t stride;
    size_t padding;
    size_t out_padding;
    size_t dilation;
};

struct ConvTranspose2dParams {
    size_t w_out;
    size_t h_out;
    size_t stride;
    size_t padding;
    size_t out_padding;
    size_t dilation;
};

struct Conv1dDepthwiseParams {
    size_t dst_numel;
    size_t l_out;
    size_t k_size;
    size_t stride;
    size_t padding;
    size_t dilation;
};

struct Conv1dDepthwiseKParams {
    size_t dst_numel;
    size_t l_out;
    size_t padding;
};

// Kernels
//
// One body per kernel, two entry points around it. The classical wrapper binds
// its scalars with `setBytes` exactly as before; the `_packed` one reads them
// from a single `device const Params*`. Neither the arithmetic nor the loop
// structure is duplicated, so the two styles cannot compute different things --
// which is what makes the bit-identical parity test meaningful rather than
// merely reassuring.
//
// Nothing in this file declares threadgroup memory, so the whole-body-plus-two-
// thin-wrappers factoring that #38 could not use for `reduce.metal` does work
// here. (MSL permits a `threadgroup` variable only inside a `[[kernel]]`-
// qualified function; there is no such variable in any of these ten families.)
//
// The array pointers are templated on the pointer *type* rather than written
// once per address space: the classical entry points pass `constant size_t *`,
// the packed ones `device const size_t *`. Deducing the address space from the
// pointer type is what lets one body serve both, and it is the same move #38
// made for `reduce.metal`'s `strided_indexer`.

template <typename T, typename PtrT>
METAL_FUNC void im2col_body(
    thread const Im2colParams &p,
    PtrT src_dims,
    PtrT src_strides,
    device const T *src,
    device T *dst,
    uint tid
) {
  // dst: (b_size, h_out, w_out, c_in, h_k, w_k)
  // src: (b_size, c_in, h_in, w_in)
  const size_t dst_numel = p.dst_numel;
  const size_t h_out = p.h_out;
  const size_t w_out = p.w_out;
  const size_t h_k = p.h_k;
  const size_t w_k = p.w_k;
  const size_t stride = p.stride;
  const size_t padding = p.padding;
  const size_t dilation = p.dilation;
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];

  const size_t dst_s4 = w_k;
  const size_t dst_s3 = h_k * dst_s4;
  const size_t dst_s2 = c_in * dst_s3;
  const size_t dst_s1 = w_out * dst_s2;
  const size_t dst_s0 = h_out * dst_s1;

  size_t tmp_tid = tid;
  const size_t b_idx = tmp_tid / dst_s0;
  tmp_tid -= b_idx * dst_s0;
  const size_t h_idx = tmp_tid / dst_s1;
  tmp_tid -= h_idx * dst_s1;
  const size_t w_idx = tmp_tid / dst_s2;
  tmp_tid -= w_idx * dst_s2;
  const size_t c_idx = tmp_tid / dst_s3;
  tmp_tid -= c_idx * dst_s3;
  const size_t h_k_idx = tmp_tid / dst_s4;
  tmp_tid -= h_k_idx * dst_s4;
  const size_t w_k_idx = tmp_tid;
  size_t src_h_idx = h_idx * stride + h_k_idx * dilation;
  size_t src_w_idx = w_idx * stride + w_k_idx * dilation;
  if (src_h_idx < padding || src_h_idx >= h_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_h_idx -= padding;
    src_w_idx -= padding;
    const size_t src_i =
      b_idx * src_strides[0]
      + c_idx * src_strides[1]
      + src_h_idx * src_strides[2]
      + src_w_idx * src_strides[3];
    dst[tid] = src[src_i];
  }
}

template <typename T>
[[kernel]] void im2col(
    constant size_t &dst_numel,
    constant size_t &h_out,
    constant size_t &w_out,
    constant size_t &h_k,
    constant size_t &w_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Im2colParams p { dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation };
  im2col_body<T, constant size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T>
[[kernel]] void im2col_packed(
    device const Im2colParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Im2colParams p = *pp;
  im2col_body<T, device const size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T>
METAL_FUNC void col2im1d_body(
    thread const Col2im1dParams &p,
    device const T *src,
    device T *dst,
    uint dst_i
) {
  // src: (b_size, l_in, c_out, l_k)
  // dst: (b_size, c_out, l_out)
  const size_t dst_el = p.dst_el;
  const size_t l_out = p.l_out;
  const size_t l_in = p.l_in;
  const size_t c_out = p.c_out;
  const size_t k_size = p.k_size;
  const size_t stride = p.stride;
  if (dst_i >= dst_el) {
    return;
  }

  const size_t dst_s0 = c_out * l_out;
  const size_t dst_s1 = l_out;
  const size_t src_s0 = c_out * k_size * l_in;
  const size_t src_s1 = c_out * k_size;
  const size_t src_s2 = k_size;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t c_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= c_idx * dst_s1;
  const int l_out_idx = tmp_dst_i;

  dst[dst_i] = static_cast<T>(0);

  int l_in_idx = l_out_idx / stride;
  int k0 = l_out_idx - l_in_idx * stride;
  // l_out_idx = l_in_idx * stride + k0
  for (; k0 < k_size && l_in_idx >= 0; k0 += stride, --l_in_idx) {
    if (l_in_idx < l_in) {
      const size_t src_i = b_idx * src_s0 + l_in_idx * src_s1 + c_idx * src_s2 + k0;
      dst[dst_i] += src[src_i];
    }
  }
}

template <typename T>
[[kernel]] void col2im1d(
    constant size_t &dst_el,
    constant size_t &l_out,
    constant size_t &l_in,
    constant size_t &c_out,
    constant size_t &k_size,
    constant size_t &stride,
    device const T *src,
    device T *dst,
    uint dst_i [[ thread_position_in_grid ]]
) {
  Col2im1dParams p { dst_el, l_out, l_in, c_out, k_size, stride };
  col2im1d_body<T>(p, src, dst, dst_i);
}

template <typename T>
[[kernel]] void col2im1d_packed(
    device const Col2im1dParams *pp,
    device const T *src,
    device T *dst,
    uint dst_i [[ thread_position_in_grid ]]
) {
  Col2im1dParams p = *pp;
  col2im1d_body<T>(p, src, dst, dst_i);
}

template <typename T, typename PtrT>
METAL_FUNC void im2col1d_body(
    thread const Im2col1dParams &p,
    PtrT src_dims,
    PtrT src_strides,
    device const T *src,
    device T *dst,
    uint tid
) {
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  const size_t dst_numel = p.dst_numel;
  const size_t l_out = p.l_out;
  const size_t l_k = p.l_k;
  const size_t stride = p.stride;
  const size_t padding = p.padding;
  const size_t dilation = p.dilation;
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t dst_s2 = l_k;
  const size_t dst_s1 = c_in * dst_s2;
  const size_t dst_s0 = l_out * dst_s1;

  size_t tmp_dst_i = tid;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const size_t c_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= c_idx * dst_s2;
  const size_t l_k_idx = tmp_dst_i;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_strides[0] + c_idx * src_strides[1] + src_l_idx * src_strides[2];
    dst[tid] = src[src_i];
  }
}

template <typename T>
[[kernel]] void im2col1d(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &l_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Im2col1dParams p { dst_numel, l_out, l_k, stride, padding, dilation };
  im2col1d_body<T, constant size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T>
[[kernel]] void im2col1d_packed(
    device const Im2col1dParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Im2col1dParams p = *pp;
  im2col1d_body<T, device const size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T, typename PtrT>
METAL_FUNC void upsample_nearest2d_body(
    thread const UpsampleNearest2dParams &p,
    PtrT src_dims,
    PtrT src_s,
    device const T *src,
    device T *dst,
    uint tid
) {
  // src: (b_size, c_in, w_in, h_in)
  const size_t w_out = p.w_out;
  const size_t h_out = p.h_out;
  const float w_scale = p.w_scale;
  const float h_scale = p.h_scale;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[tid] = src[src_i];
}

template <typename T>
[[kernel]] void upsample_nearest2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant float &w_scale,
    constant float &h_scale,
    constant size_t *src_dims,
    constant size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  UpsampleNearest2dParams p { w_out, h_out, w_scale, h_scale };
  upsample_nearest2d_body<T, constant size_t *>(p, src_dims, src_s, src, dst, tid);
}

template <typename T>
[[kernel]] void upsample_nearest2d_packed(
    device const UpsampleNearest2dParams *pp,
    device const size_t *src_dims,
    device const size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  UpsampleNearest2dParams p = *pp;
  upsample_nearest2d_body<T, device const size_t *>(p, src_dims, src_s, src, dst, tid);
}

template <typename T, typename PtrT>
METAL_FUNC void upsample_bilinear2d_body(
    thread const UpsampleBilinear2dParams &p,
    PtrT src_dims,
    PtrT src_s,
    device const T *src,
    device T *dst,
    uint tid
) {
    const size_t w_out = p.w_out;
    const size_t h_out = p.h_out;
    const bool align_corners = p.align_corners;
    const bool has_scale_h = p.has_scale_h;
    const float scale_h_factor = p.scale_h_factor;
    const bool has_scale_w = p.has_scale_w;
    const float scale_w_factor = p.scale_w_factor;

    // src: (b_size, c_in, h_in, w_in)  // Standard NCHW layout
    const size_t c = src_dims[1];
    const size_t h_in = src_dims[2];  // dims[2] = height
    const size_t w_in = src_dims[3];  // dims[3] = width
    
    if (tid >= src_dims[0] * c * h_out * w_out) {
        return;
    }
    
    // Compute output position (NCHW layout)
    const size_t b_idx = tid / (h_out * w_out * c);
    const size_t c_idx = (tid / (h_out * w_out)) % c;
    const size_t dst_h = (tid / w_out) % h_out;
    const size_t dst_w = tid % w_out;
    
    // Calculate scale factors following PyTorch's area_pixel_compute_scale logic
    float h_scale, w_scale;
    if (align_corners) {
        h_scale = (h_out > 1) ? static_cast<float>(h_in - 1) / (h_out - 1) : 0.0f;
        w_scale = (w_out > 1) ? static_cast<float>(w_in - 1) / (w_out - 1) : 0.0f;
    } else {
        // PyTorch's compute_scales_value logic
        h_scale = has_scale_h ? (1.0f / scale_h_factor) : (static_cast<float>(h_in) / h_out);
        w_scale = has_scale_w ? (1.0f / scale_w_factor) : (static_cast<float>(w_in) / w_out);
    }
    
    // Compute source position
    float src_h_fp, src_w_fp;
    if (align_corners) {
        src_h_fp = h_scale * dst_h;
        src_w_fp = w_scale * dst_w;
    } else {
        src_h_fp = h_scale * (dst_h + 0.5f) - 0.5f;
        src_w_fp = w_scale * (dst_w + 0.5f) - 0.5f;
    }
    
    // Clamp to valid range
    src_h_fp = max(0.0f, src_h_fp);
    src_w_fp = max(0.0f, src_w_fp);
    
    // Get integer indices
    size_t h0 = static_cast<size_t>(floor(src_h_fp));
    size_t w0 = static_cast<size_t>(floor(src_w_fp));
    size_t h1 = min(h0 + 1, h_in - 1);
    size_t w1 = min(w0 + 1, w_in - 1);
    
    // Compute interpolation weights
    float weight_h = src_h_fp - h0;
    float weight_w = src_w_fp - w0;
    weight_h = clamp(weight_h, 0.0f, 1.0f);
    weight_w = clamp(weight_w, 0.0f, 1.0f);
    
    // Get base index
    const size_t base = b_idx * src_s[0] + c_idx * src_s[1];
    
    // Read four neighboring pixels
    const T v00 = src[base + h0 * src_s[2] + w0 * src_s[3]];
    const T v10 = src[base + h0 * src_s[2] + w1 * src_s[3]];
    const T v01 = src[base + h1 * src_s[2] + w0 * src_s[3]];
    const T v11 = src[base + h1 * src_s[2] + w1 * src_s[3]];
    
    // Bilinear interpolation
    const float v_top = float(v00) * (1.0f - weight_w) + float(v10) * weight_w;
    const float v_bottom = float(v01) * (1.0f - weight_w) + float(v11) * weight_w;
    const float value = v_top * (1.0f - weight_h) + v_bottom * weight_h;
    
    dst[tid] = T(value);
}

// Buffer indices are explicit here because the macro that used to generate this
// kernel's signature spelled them out; keeping them pinned means the binding
// layout stays fixed even if a parameter is added above.
template <typename T>
[[kernel]] void upsample_bilinear2d(
    constant size_t &w_out [[buffer(0)]],
    constant size_t &h_out [[buffer(1)]],
    constant bool &align_corners [[buffer(2)]],
    constant bool &has_scale_h [[buffer(3)]],
    constant float &scale_h_factor [[buffer(4)]],
    constant bool &has_scale_w [[buffer(5)]],
    constant float &scale_w_factor [[buffer(6)]],
    constant size_t *src_dims [[buffer(7)]],
    constant size_t *src_s [[buffer(8)]],
    device const T *src [[buffer(9)]],
    device T *dst [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    UpsampleBilinear2dParams p {
        w_out, h_out, align_corners, has_scale_h, scale_h_factor,
        has_scale_w, scale_w_factor
    };
    upsample_bilinear2d_body<T, constant size_t *>(p, src_dims, src_s, src, dst, tid);
}

// The packed indices are contiguous from 0 rather than carrying the classical
// pins forward: seven scalars leave the argument list, so slots 0..6 are gone
// and everything after them moves down. That renumbering is exactly what
// `ParamCapture` performs on the Rust side, and stating it explicitly here keeps
// the two descriptions of one layout next to each other. Getting it wrong is
// silent under `HazardTrackingModeUntracked` (`DESIGN.md` §3.5), which is what
// the bit-identical parity test exists to catch.
template <typename T>
[[kernel]] void upsample_bilinear2d_packed(
    device const UpsampleBilinear2dParams *pp [[buffer(0)]],
    device const size_t *src_dims [[buffer(1)]],
    device const size_t *src_s [[buffer(2)]],
    device const T *src [[buffer(3)]],
    device T *dst [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    UpsampleBilinear2dParams p = *pp;
    upsample_bilinear2d_body<T, device const size_t *>(p, src_dims, src_s, src, dst, tid);
}

template <typename T, typename A, typename PtrT>
METAL_FUNC void avg_pool2d_body(
    thread const Pool2dParams &p,
    PtrT src_dims,
    PtrT src_strides,
    device const T *src,
    device T *dst,
    uint tid
) {
  const size_t w_k = p.w_k;
  const size_t h_k = p.h_k;
  const size_t w_stride = p.w_stride;
  const size_t h_stride = p.h_stride;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[tid] = static_cast<T>(d / (w_k * h_k));
}

// `A` is the accumulator and stays a template parameter on both wrappers. The
// integer instantiations accumulate in their own type rather than widening, so
// their averaging truncates exactly where it did before -- a binding change must
// not move that (`DESIGN.md` §8.1c).
template <typename T, typename A>
[[kernel]] void avg_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Pool2dParams p { w_k, h_k, w_stride, h_stride };
  avg_pool2d_body<T, A, constant size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T, typename A>
[[kernel]] void avg_pool2d_packed(
    device const Pool2dParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Pool2dParams p = *pp;
  avg_pool2d_body<T, A, device const size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T, typename PtrT>
METAL_FUNC void max_pool2d_body(
    thread const Pool2dParams &p,
    PtrT src_dims,
    PtrT src_strides,
    device const T *src,
    device T *dst,
    uint tid
) {
  const size_t w_k = p.w_k;
  const size_t h_k = p.h_k;
  const size_t w_stride = p.w_stride;
  const size_t h_stride = p.h_stride;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  T d = 0;
  bool set = false;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      if (set) {
        d = MAX(d, src[src_idx]);
      }
      else {
        d = src[src_idx];
        set = true;
      }
    }
  }
  dst[tid] = d;
}

template <typename T>
[[kernel]] void max_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Pool2dParams p { w_k, h_k, w_stride, h_stride };
  max_pool2d_body<T, constant size_t *>(p, src_dims, src_strides, src, dst, tid);
}

template <typename T>
[[kernel]] void max_pool2d_packed(
    device const Pool2dParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Pool2dParams p = *pp;
  max_pool2d_body<T, device const size_t *>(p, src_dims, src_strides, src, dst, tid);
}


// Naive implementation of conv_transpose1d.
template <typename T, typename A, typename PtrT>
METAL_FUNC void conv_transpose1d_body(
    thread const ConvTranspose1dParams &p,
    PtrT src_dims,
    PtrT src_strides,
    PtrT k_dims,
    PtrT k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid
) {
  // src: (b_size, c_in, l_in)
  // kernel: (c_in, c_out, l_k)
  const size_t l_out = p.l_out;
  const size_t stride = p.stride;
  const size_t padding = p.padding;
  const size_t dilation = p.dilation;
  const size_t l_k = k_dims[2];
  const size_t c_out = k_dims[1];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (tid >= src_dims[0] * c_out * l_out) {
    return;
  }

  const size_t b_idx = tid / (l_out * c_out);
  const size_t dst_c_idx = (tid / l_out) % c_out;
  const size_t out_x = tid % l_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
      // let out_x = inp_x * p.stride + k_x * p.dilation - p.padding;
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      int inp_x = inp_x_stride / stride;
      if (inp_x >= l_in) continue;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
          const size_t src_idx = src_idx0 + src_c_idx * src_strides[1] + inp_x * src_strides[2];
          const size_t k_idx = src_c_idx * k_strides[0] + dst_c_idx * k_strides[1] + k_x * k_strides[2];
          d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
      }
  }
  dst[tid] = static_cast<T>(d);
}

// `out_padding` is bound but never read by the body, here and in
// `conv_transpose2d`. It stays a field rather than being dropped: the packed
// block is built by letting the existing `set_params!` call run and diverting
// each scalar as it passes, so the struct must mirror the argument list exactly
// or every field after the omission lands at the wrong offset.
template <typename T, typename A>
[[kernel]] void conv_transpose1d(
    constant size_t &l_out,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &out_padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    constant size_t *k_dims,
    constant size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  ConvTranspose1dParams p { l_out, stride, padding, out_padding, dilation };
  conv_transpose1d_body<T, A, constant size_t *>(
      p, src_dims, src_strides, k_dims, k_strides, src, k, dst, tid);
}

template <typename T, typename A>
[[kernel]] void conv_transpose1d_packed(
    device const ConvTranspose1dParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const size_t *k_dims,
    device const size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  ConvTranspose1dParams p = *pp;
  conv_transpose1d_body<T, A, device const size_t *>(
      p, src_dims, src_strides, k_dims, k_strides, src, k, dst, tid);
}

template <typename T, typename A, typename PtrT>
METAL_FUNC void conv_transpose2d_body(
  thread const ConvTranspose2dParams &p,
  PtrT input_dims,
  PtrT input_stride,
  PtrT k_dims,
  PtrT k_stride,
  device const T *src,
  device const T *k,
  device T *dst,
  uint tid
) {
  const size_t w_out = p.w_out;
  const size_t h_out = p.h_out;
  const size_t stride = p.stride;
  const size_t padding = p.padding;
  const size_t dilation = p.dilation;
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out = k_dims[1];
  const size_t c_in = input_dims[1];
  const size_t h_in = input_dims[2];
  const size_t w_in = input_dims[3];

  if (tid >= input_dims[0] * c_out * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c_out);
  const size_t dst_c_idx = (tid / (w_out * h_out)) % c_out;
  const size_t out_y = (tid / w_out) % h_out;
  const size_t out_x = tid % w_out;

  const size_t src_idx0 = b_idx * input_stride[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)w_k; ++k_x) {
      const int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      const int inp_x = inp_x_stride / stride;
      if (inp_x >= w_in) continue;
      for (int k_y = 0; k_y < (int)h_k; ++k_y) {
          const int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
          if (inp_y_stride < 0 || inp_y_stride % stride) {
              continue;
          }
          const int inp_y = inp_y_stride / stride;
          if (inp_y >= h_in) continue;
          for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
              const size_t src_idx = src_idx0 + src_c_idx * input_stride[1] + inp_y * input_stride[2] + inp_x * input_stride[3];
              const size_t k_idx = src_c_idx * k_stride[0] + dst_c_idx * k_stride[1] + k_y * k_stride[2] + k_x * k_stride[3];
              d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
          }
      }
  }
  dst[tid] = static_cast<T>(d);
}

template <typename T, typename A>
[[kernel]] void conv_transpose2d(
  constant size_t &w_out,
  constant size_t &h_out,
  constant size_t &stride,
  constant size_t &padding,
  constant size_t &out_padding,
  constant size_t &dilation,
  constant size_t *input_dims,
  constant size_t *input_stride,
  constant size_t *k_dims,
  constant size_t *k_stride,
  device const T *src,
  device const T *k,
  device T *dst,
  uint tid [[ thread_position_in_grid ]]
) {
  ConvTranspose2dParams p { w_out, h_out, stride, padding, out_padding, dilation };
  conv_transpose2d_body<T, A, constant size_t *>(
      p, input_dims, input_stride, k_dims, k_stride, src, k, dst, tid);
}

template <typename T, typename A>
[[kernel]] void conv_transpose2d_packed(
  device const ConvTranspose2dParams *pp,
  device const size_t *input_dims,
  device const size_t *input_stride,
  device const size_t *k_dims,
  device const size_t *k_stride,
  device const T *src,
  device const T *k,
  device T *dst,
  uint tid [[ thread_position_in_grid ]]
) {
  ConvTranspose2dParams p = *pp;
  conv_transpose2d_body<T, A, device const size_t *>(
      p, input_dims, input_stride, k_dims, k_stride, src, k, dst, tid);
}

// Depthwise 1D convolution, fused.
//
// The generic path builds an im2col matrix, runs a matmul, then transposes the
// result into place — three dispatches and two intermediate buffers. For a
// depthwise conv (groups == c_in, so each output channel reads only its own
// input channel) the matmul is degenerate: every output is a short dot product
// of k_size taps. candle also splits grouped convs into one conv per group and
// concatenates, so a 2048-channel depthwise layer becomes thousands of
// dispatches. This does the whole layer in one, writing straight to the final
// (b, c, l_out) layout so no transpose copy is needed.
template <typename T, typename A, typename PtrT>
METAL_FUNC void conv1d_depthwise_body(
    thread const Conv1dDepthwiseParams &p,
    PtrT src_dims,
    PtrT src_strides,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid
) {
  const size_t dst_numel = p.dst_numel;
  const size_t l_out = p.l_out;
  const size_t k_size = p.k_size;
  const size_t stride = p.stride;
  const size_t padding = p.padding;
  const size_t dilation = p.dilation;
  if (tid >= dst_numel) {
    return;
  }

  const size_t c = src_dims[1];
  const size_t l_in = src_dims[2];

  // dst is contiguous (b, c, l_out); recover the index directly.
  const size_t l_idx = tid % l_out;
  const size_t tmp = tid / l_out;
  const size_t c_idx = tmp % c;
  const size_t b_idx = tmp / c;

  // Accumulate in a wider type: f16 sums lose precision quickly, and this
  // costs nothing since the taps are few.
  A acc = static_cast<A>(0);
  const size_t src_base = b_idx * src_strides[0] + c_idx * src_strides[1];
  const size_t w_base = c_idx * k_size;

  for (size_t k = 0; k < k_size; ++k) {
    const size_t pos = l_idx * stride + k * dilation;
    // Positions inside the padding contribute nothing.
    if (pos < padding) {
      continue;
    }
    const size_t src_l = pos - padding;
    if (src_l >= l_in) {
      continue;
    }
    acc += static_cast<A>(src[src_base + src_l * src_strides[2]])
         * static_cast<A>(weight[w_base + k]);
  }

  dst[tid] = static_cast<T>(acc);
}

template <typename T, typename A>
[[kernel]] void conv1d_depthwise(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &k_size,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Conv1dDepthwiseParams p { dst_numel, l_out, k_size, stride, padding, dilation };
  conv1d_depthwise_body<T, A, constant size_t *>(
      p, src_dims, src_strides, src, weight, dst, tid);
}

template <typename T, typename A>
[[kernel]] void conv1d_depthwise_packed(
    device const Conv1dDepthwiseParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Conv1dDepthwiseParams p = *pp;
  conv1d_depthwise_body<T, A, device const size_t *>(
      p, src_dims, src_strides, src, weight, dst, tid);
}

// Depthwise 1D convolution, specialized for the common contiguous case.
//
// Same computation as `conv1d_depthwise` above, with three things moved from
// runtime to compile time:
//
//   - K_SIZE is a template parameter, so the tap loop has a constant trip count
//     and unrolls. The generic kernel reads k_size from a buffer, so it cannot.
//   - stride and dilation are fixed at 1, so `pos` is `l_idx + k` — one add,
//     no multiplies.
//   - the source is contiguous, so addressing is `(b*c + c_idx)*l_in + src_l`
//     rather than three loads from src_strides[] and three multiplies.
//
// The padding branches stay, but they are now comparisons against compile-time
// K_SIZE-derived values inside an unrolled body, so the compiler hoists what it
// can. They are *not* eliminated: `l_idx` is a runtime value, so which taps fall
// in the padding genuinely varies per thread. See the measurement note in the
// issue-10 write-up — the branches diverge only in the simdgroups that straddle
// an l_out boundary, which for a long prefill is a small minority.
//
// Preconditions the caller must check (see `call_conv1d_depthwise_k`):
// stride == 1, dilation == 1, contiguous source, and K_SIZE matching an
// instantiated variant. Anything else uses the generic kernel above.
template <typename T, typename A, ushort K_SIZE, typename PtrT>
METAL_FUNC void conv1d_depthwise_k_body(
    thread const Conv1dDepthwiseKParams &p,
    PtrT src_dims,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid
) {
  const size_t dst_numel = p.dst_numel;
  const size_t l_out = p.l_out;
  const size_t padding = p.padding;
  if (tid >= dst_numel) {
    return;
  }

  const size_t c = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t l_idx = tid % l_out;
  const size_t tmp = tid / l_out;
  const size_t c_idx = tmp % c;
  const size_t b_idx = tmp / c;

  // Accumulate in a wider type for the same reason the generic kernel does.
  A acc = static_cast<A>(0);
  // Contiguous (b, c, l_in): the row base is a single multiply-add.
  const size_t src_base = (b_idx * c + c_idx) * l_in;
  const size_t w_base = c_idx * K_SIZE;

#pragma clang loop unroll(full)
  for (ushort k = 0; k < K_SIZE; ++k) {
    // stride == 1 and dilation == 1 are preconditions, so this is just l_idx + k.
    const size_t pos = l_idx + k;
    if (pos < padding) {
      continue;
    }
    const size_t src_l = pos - padding;
    if (src_l >= l_in) {
      continue;
    }
    acc += static_cast<A>(src[src_base + src_l])
         * static_cast<A>(weight[w_base + k]);
  }

  dst[tid] = static_cast<T>(acc);
}

// `K_SIZE` stays a template parameter on both wrappers -- it is a compile-tier
// axis (`DESIGN.md` §7.4) that fixes the tap loop's trip count so it unrolls,
// which is the whole point of this variant. The binding style is a *second*,
// independent axis, and the two compose rather than interacting.
template <typename T, typename A, ushort K_SIZE>
[[kernel]] void conv1d_depthwise_k(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &padding,
    constant size_t *src_dims,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Conv1dDepthwiseKParams p { dst_numel, l_out, padding };
  conv1d_depthwise_k_body<T, A, K_SIZE, constant size_t *>(
      p, src_dims, src, weight, dst, tid);
}

template <typename T, typename A, ushort K_SIZE>
[[kernel]] void conv1d_depthwise_k_packed(
    device const Conv1dDepthwiseKParams *pp,
    device const size_t *src_dims,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  Conv1dDepthwiseKParams p = *pp;
  conv1d_depthwise_k_body<T, A, K_SIZE, device const size_t *>(
      p, src_dims, src, weight, dst, tid);
}

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `conv_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
static_assert(sizeof(Im2colParams) == 64, "Im2colParams layout");
static_assert(alignof(Im2colParams) == 8, "Im2colParams alignment");

static_assert(sizeof(Col2im1dParams) == 48, "Col2im1dParams layout");
static_assert(alignof(Col2im1dParams) == 8, "Col2im1dParams alignment");

static_assert(sizeof(Im2col1dParams) == 48, "Im2col1dParams layout");
static_assert(alignof(Im2col1dParams) == 8, "Im2col1dParams alignment");

static_assert(sizeof(UpsampleNearest2dParams) == 24, "UpsampleNearest2dParams layout");
static_assert(alignof(UpsampleNearest2dParams) == 8, "UpsampleNearest2dParams alignment");

// The mixed-width one. Three `bool` at 1 byte each and two `float` between two
// `size_t`: `align_corners` and `has_scale_h` sit adjacent at 16 and 17, then
// `scale_h_factor` pads to 20, `has_scale_w` lands at 24, `scale_w_factor` pads
// to 28, and the struct pads up to its own 8-byte alignment at 32. Every one of
// those numbers is a padding rule rather than a field width, which is why the
// cross-boundary check matters more here than anywhere else in the file.
static_assert(sizeof(UpsampleBilinear2dParams) == 32, "UpsampleBilinear2dParams layout");
static_assert(alignof(UpsampleBilinear2dParams) == 8, "UpsampleBilinear2dParams alignment");

static_assert(sizeof(Pool2dParams) == 32, "Pool2dParams layout");
static_assert(alignof(Pool2dParams) == 8, "Pool2dParams alignment");

static_assert(sizeof(ConvTranspose1dParams) == 40, "ConvTranspose1dParams layout");
static_assert(alignof(ConvTranspose1dParams) == 8, "ConvTranspose1dParams alignment");

static_assert(sizeof(ConvTranspose2dParams) == 48, "ConvTranspose2dParams layout");
static_assert(alignof(ConvTranspose2dParams) == 8, "ConvTranspose2dParams alignment");

static_assert(sizeof(Conv1dDepthwiseParams) == 48, "Conv1dDepthwiseParams layout");
static_assert(alignof(Conv1dDepthwiseParams) == 8, "Conv1dDepthwiseParams alignment");

static_assert(sizeof(Conv1dDepthwiseKParams) == 24, "Conv1dDepthwiseKParams layout");
static_assert(alignof(Conv1dDepthwiseKParams) == 8, "Conv1dDepthwiseKParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void conv_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    Im2colParams              probe_Im2colParams;
    Col2im1dParams            probe_Col2im1dParams;
    Im2col1dParams            probe_Im2col1dParams;
    UpsampleNearest2dParams   probe_UpsampleNearest2dParams;
    UpsampleBilinear2dParams  probe_UpsampleBilinear2dParams;
    Pool2dParams              probe_Pool2dParams;
    ConvTranspose1dParams     probe_ConvTranspose1dParams;
    ConvTranspose2dParams     probe_ConvTranspose2dParams;
    Conv1dDepthwiseParams     probe_Conv1dDepthwiseParams;
    Conv1dDepthwiseKParams    probe_Conv1dDepthwiseKParams;

    out[0]  = sizeof(Im2colParams);
    out[1]  = offsetof_rt(Im2colParams, dst_numel);
    out[2]  = offsetof_rt(Im2colParams, h_out);
    out[3]  = offsetof_rt(Im2colParams, w_out);
    out[4]  = offsetof_rt(Im2colParams, h_k);
    out[5]  = offsetof_rt(Im2colParams, w_k);
    out[6]  = offsetof_rt(Im2colParams, stride);
    out[7]  = offsetof_rt(Im2colParams, padding);
    out[8]  = offsetof_rt(Im2colParams, dilation);

    out[9]  = sizeof(Col2im1dParams);
    out[10] = offsetof_rt(Col2im1dParams, dst_el);
    out[11] = offsetof_rt(Col2im1dParams, l_out);
    out[12] = offsetof_rt(Col2im1dParams, l_in);
    out[13] = offsetof_rt(Col2im1dParams, c_out);
    out[14] = offsetof_rt(Col2im1dParams, k_size);
    out[15] = offsetof_rt(Col2im1dParams, stride);

    out[16] = sizeof(Im2col1dParams);
    out[17] = offsetof_rt(Im2col1dParams, dst_numel);
    out[18] = offsetof_rt(Im2col1dParams, l_out);
    out[19] = offsetof_rt(Im2col1dParams, l_k);
    out[20] = offsetof_rt(Im2col1dParams, stride);
    out[21] = offsetof_rt(Im2col1dParams, padding);
    out[22] = offsetof_rt(Im2col1dParams, dilation);

    out[23] = sizeof(UpsampleNearest2dParams);
    out[24] = offsetof_rt(UpsampleNearest2dParams, w_out);
    out[25] = offsetof_rt(UpsampleNearest2dParams, h_out);
    out[26] = offsetof_rt(UpsampleNearest2dParams, w_scale);
    out[27] = offsetof_rt(UpsampleNearest2dParams, h_scale);

    out[28] = sizeof(UpsampleBilinear2dParams);
    out[29] = offsetof_rt(UpsampleBilinear2dParams, w_out);
    out[30] = offsetof_rt(UpsampleBilinear2dParams, h_out);
    out[31] = offsetof_rt(UpsampleBilinear2dParams, align_corners);
    out[32] = offsetof_rt(UpsampleBilinear2dParams, has_scale_h);
    out[33] = offsetof_rt(UpsampleBilinear2dParams, scale_h_factor);
    out[34] = offsetof_rt(UpsampleBilinear2dParams, has_scale_w);
    out[35] = offsetof_rt(UpsampleBilinear2dParams, scale_w_factor);

    out[36] = sizeof(Pool2dParams);
    out[37] = offsetof_rt(Pool2dParams, w_k);
    out[38] = offsetof_rt(Pool2dParams, h_k);
    out[39] = offsetof_rt(Pool2dParams, w_stride);
    out[40] = offsetof_rt(Pool2dParams, h_stride);

    out[41] = sizeof(ConvTranspose1dParams);
    out[42] = offsetof_rt(ConvTranspose1dParams, l_out);
    out[43] = offsetof_rt(ConvTranspose1dParams, stride);
    out[44] = offsetof_rt(ConvTranspose1dParams, padding);
    out[45] = offsetof_rt(ConvTranspose1dParams, out_padding);
    out[46] = offsetof_rt(ConvTranspose1dParams, dilation);

    out[47] = sizeof(ConvTranspose2dParams);
    out[48] = offsetof_rt(ConvTranspose2dParams, w_out);
    out[49] = offsetof_rt(ConvTranspose2dParams, h_out);
    out[50] = offsetof_rt(ConvTranspose2dParams, stride);
    out[51] = offsetof_rt(ConvTranspose2dParams, padding);
    out[52] = offsetof_rt(ConvTranspose2dParams, out_padding);
    out[53] = offsetof_rt(ConvTranspose2dParams, dilation);

    out[54] = sizeof(Conv1dDepthwiseParams);
    out[55] = offsetof_rt(Conv1dDepthwiseParams, dst_numel);
    out[56] = offsetof_rt(Conv1dDepthwiseParams, l_out);
    out[57] = offsetof_rt(Conv1dDepthwiseParams, k_size);
    out[58] = offsetof_rt(Conv1dDepthwiseParams, stride);
    out[59] = offsetof_rt(Conv1dDepthwiseParams, padding);
    out[60] = offsetof_rt(Conv1dDepthwiseParams, dilation);

    out[61] = sizeof(Conv1dDepthwiseKParams);
    out[62] = offsetof_rt(Conv1dDepthwiseKParams, dst_numel);
    out[63] = offsetof_rt(Conv1dDepthwiseKParams, l_out);
    out[64] = offsetof_rt(Conv1dDepthwiseKParams, padding);
}

// Explicit instantiation. `decltype(func<...>)` restates the template's own
// signature, so a variant is declared by naming the type arguments and the
// `[[host_name]]` string only — the parameter list is written once, in the
// template. The macro-per-family form this replaces spelled every signature
// twice, and the two could drift silently.
//
// Same spelling as unary.metal, binary.metal and affine.metal, which candle
// already migrated.
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both binding styles from one instantiation row, so a variant cannot exist in
// one style and not the other. `_packed` is a name segment appended after the
// dtype and any `_k<K>`, and `packed_names_resolve` checks every result against
// the compiled library rather than against these macros -- which is
// `DESIGN.md` §8.1b's argument, and #26 shipped 48 names absent from a metallib
// that compiled cleanly.
//
// bfloat is gated per family exactly as before this conversion: the depthwise
// kernel on __METAL_VERSION__ >= 310, every other family on __HAVE_BFLOAT__.
// The two guards are not interchangeable, so they are kept as they were rather
// than unified. Because each row emits both styles, the guards carry the packed
// variants with them and the two styles cannot come apart per dtype.
#define init_conv1d_depthwise(tname, t, acc) \
    init_kernel("conv1d_depthwise_" #tname, conv1d_depthwise, t, acc) \
    init_kernel("conv1d_depthwise_" #tname "_packed", conv1d_depthwise_packed, t, acc)

// The specialized variant carries k_size in its name, per DESIGN.md §7.4's
// `<op>_<dtype>_k<K>` shape. K is a compile-tier axis: it fixes the tap loop's
// trip count, so it changes the generated code and the register allocation.
// The binding style is a second, independent compile-tier axis, and `_packed`
// follows `_k<K>` so the name reads outermost-axis-last.
#define init_conv1d_depthwise_k(tname, t, acc, k) \
    init_kernel("conv1d_depthwise_" #tname "_k" #k, conv1d_depthwise_k, t, acc, k) \
    init_kernel("conv1d_depthwise_" #tname "_k" #k "_packed", conv1d_depthwise_k_packed, t, acc, k)

#define init_im2col1d(tname, t) \
    init_kernel("im2col1d_" #tname, im2col1d, t) \
    init_kernel("im2col1d_" #tname "_packed", im2col1d_packed, t)

#define init_im2col(tname, t) \
    init_kernel("im2col_" #tname, im2col, t) \
    init_kernel("im2col_" #tname "_packed", im2col_packed, t)

#define init_col2im1d(tname, t) \
    init_kernel("col2im1d_" #tname, col2im1d, t) \
    init_kernel("col2im1d_" #tname "_packed", col2im1d_packed, t)

#define init_upsample_nearest2d(tname, t) \
    init_kernel("upsample_nearest2d_" #tname, upsample_nearest2d, t) \
    init_kernel("upsample_nearest2d_" #tname "_packed", upsample_nearest2d_packed, t)

#define init_upsample_bilinear2d(tname, t) \
    init_kernel("upsample_bilinear2d_" #tname, upsample_bilinear2d, t) \
    init_kernel("upsample_bilinear2d_" #tname "_packed", upsample_bilinear2d_packed, t)

#define init_max_pool2d(tname, t) \
    init_kernel("max_pool2d_" #tname, max_pool2d, t) \
    init_kernel("max_pool2d_" #tname "_packed", max_pool2d_packed, t)

#define init_avg_pool2d(tname, t, acc) \
    init_kernel("avg_pool2d_" #tname, avg_pool2d, t, acc) \
    init_kernel("avg_pool2d_" #tname "_packed", avg_pool2d_packed, t, acc)

#define init_conv_transpose1d(tname, t, acc) \
    init_kernel("conv_transpose1d_" #tname, conv_transpose1d, t, acc) \
    init_kernel("conv_transpose1d_" #tname "_packed", conv_transpose1d_packed, t, acc)

#define init_conv_transpose2d(tname, t, acc) \
    init_kernel("conv_transpose2d_" #tname, conv_transpose2d, t, acc) \
    init_kernel("conv_transpose2d_" #tname "_packed", conv_transpose2d_packed, t, acc)

init_im2col(f32, float);
init_im2col(f16, half);
init_im2col(u8, uint8_t);
init_im2col(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_im2col(bf16, bfloat);
#endif

init_col2im1d(f32, float);
init_col2im1d(f16, half);
init_col2im1d(u8, uint8_t);
init_col2im1d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_col2im1d(bf16, bfloat);
#endif

init_conv1d_depthwise(f32, float, float);
init_conv1d_depthwise(f16, half, float);
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
init_conv1d_depthwise(bf16, bfloat, float);
#endif

// k = 2, 3, 4 covers the depthwise widths that occur in practice; LFM2 uses 3
// (conv_L_cache). Anything outside the set falls back to the generic kernel
// above, so the list is a performance choice and never a correctness one.
init_conv1d_depthwise_k(f32, float, float, 2);
init_conv1d_depthwise_k(f32, float, float, 3);
init_conv1d_depthwise_k(f32, float, float, 4);
init_conv1d_depthwise_k(f16, half, float, 2);
init_conv1d_depthwise_k(f16, half, float, 3);
init_conv1d_depthwise_k(f16, half, float, 4);
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
init_conv1d_depthwise_k(bf16, bfloat, float, 2);
init_conv1d_depthwise_k(bf16, bfloat, float, 3);
init_conv1d_depthwise_k(bf16, bfloat, float, 4);
#endif

init_im2col1d(f32, float);
init_im2col1d(f16, half);
init_im2col1d(u8, uint8_t);
init_im2col1d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_im2col1d(bf16, bfloat);
#endif

init_upsample_nearest2d(f32, float);
init_upsample_nearest2d(f16, half);
init_upsample_nearest2d(u8, uint8_t);
init_upsample_nearest2d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_upsample_nearest2d(bf16, bfloat);
#endif

init_upsample_bilinear2d(f32, float);
init_upsample_bilinear2d(f16, half);
init_upsample_bilinear2d(u8, uint8_t);
init_upsample_bilinear2d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_upsample_bilinear2d(bf16, bfloat);
#endif

init_max_pool2d(f32, float);
init_max_pool2d(f16, half);
init_max_pool2d(u32, uint32_t);
init_max_pool2d(u8, uint8_t);
#if defined(__HAVE_BFLOAT__)
init_max_pool2d(bf16, bfloat);
#endif

// Accumulator types differ per dtype and are load-bearing: the integer
// instantiations accumulate in their own type, so their averaging truncates
// exactly as it did before. Widening them would be a behaviour change.
init_avg_pool2d(f32, float, float);
init_avg_pool2d(f16, half, float);
init_avg_pool2d(u32, uint32_t, uint32_t);
init_avg_pool2d(u8, uint8_t, uint8_t);
#if defined(__HAVE_BFLOAT__)
init_avg_pool2d(bf16, bfloat, float);
#endif

init_conv_transpose1d(f32, float, float);
init_conv_transpose1d(f16, half, float);
init_conv_transpose1d(u8, uint8_t, uint8_t);
init_conv_transpose1d(u32, uint32_t, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_conv_transpose1d(bf16, bfloat, float);
#endif

init_conv_transpose2d(f32, float, float);
init_conv_transpose2d(f16, half, float);
#if defined(__HAVE_BFLOAT__)
init_conv_transpose2d(bf16, bfloat, float);
#endif
