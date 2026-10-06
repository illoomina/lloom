use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::kernels::metal::{Buffer, BufferHandle};
use crate::{DType, Layout, MetalDevice, MetalStorage, Result, Shape, D};

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: BufferHandle,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("qstorage_zeros")
            .build()?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    /// A persistent, zeroed storage for `elem_count` elements of `dtype`
    /// outside every pool (`MetalDevice::new_persistent_private`): the slot
    /// pool of a streamed weight tier.
    pub fn persistent(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        Self::persistent_with(device, elem_count, dtype, false)
    }

    /// [`Self::persistent`], Shared when `shared`: the CPU writes the bytes
    /// in place (a streamed expert read straight into its slot).
    pub fn persistent_with(
        device: &MetalDevice,
        elem_count: usize,
        dtype: GgmlDType,
        shared: bool,
    ) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = if shared {
            device.new_persistent_shared(size)?
        } else {
            device.new_persistent_private(size)?
        };
        {
            let mut blit = device.blit_command_encoder()?;
            blit.set_label("qstorage_persistent_zero");
            blit.fill_buffer(&buffer, (0, size), 0);
        }
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_dequantize_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2XXS => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ2xxs> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ2xxs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2XS => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ2xs> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ2xs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2S => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ2s> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ2s::to_float(&vec, &mut out);
            }
            GgmlDType::IQ3XXS => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ3xxs> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ3xxs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ3S => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ3s> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ3s::to_float(&vec, &mut out);
            }
            GgmlDType::IQ4NL => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ4nl> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ4nl::to_float(&vec, &mut out);
            }
            GgmlDType::IQ4XS => {
                let vec: Vec<crate::quantized::iq_quants::BlockIQ4xs> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockIQ4xs::to_float(&vec, &mut out);
            }
            GgmlDType::Q2_0 => {
                let vec: Vec<crate::quantized::iq_quants::BlockQ2_0> =
                    read_to_vec(&buffer, block_len);
                crate::quantized::iq_quants::BlockQ2_0::to_float(&vec, &mut out);
            }
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&out)
            .with_label("qstorage_dequantized")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantized")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row)?;
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix_onto")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?)?;
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_onto")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<MetalStorage> {
        use crate::MetalError;

        if ids.dtype() != DType::U32 {
            crate::bail!("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        // The pool serves the smallest free bucket at or above the request
        // (§11.5n), so the buffer may be LARGER than the tensor; only a
        // buffer too small to hold `rows` is a defect.
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() < expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected at least {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids_len = ids_l.shape().elem_count();
        let device = self.device.clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        crate::kernels::call_quantized_get_rows(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            hidden,
            hidden * self.dtype.type_size() / self.dtype.block_size(),
            ids_len,
            &self.buffer,
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(
            dst,
            device.clone(),
            ids_len * hidden,
            DType::F32,
        ))
    }

    /// Rows up to which a quantized matmul runs the matmul-vec kernel over
    /// the rows instead of the mm kernel (see `fwd`); `LLOOM_QMV_MM_FROM=N`
    /// (read once) lowers the mm kernel's threshold to `N` rows — the mm
    /// kernel dequantizes a tile once for its rows (0.27 vs 0.36 ms on a
    /// q6_K `[10240 × 2560]` at 4 rows) and is another accumulation order
    /// than the row gemvs, so it is the measurement arm for a multi-row
    /// pass that gives up bit-identity with the single row.
    fn qmv_rows_max() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            crate::tuning::tuning().qmv
                .mm_from
                .map(|from| from.saturating_sub(1).max(1))
                .unwrap_or(8)
        })
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();

        // Issue #750: the activation dtype chooses the `(activation, accumulate,
        // store)` triple. An f16 activation reads and stores f16 (accumulating in
        // f32 — §8.1.4), so a quantized weight no longer forces the activation and
        // store to f32; anything else keeps the historical `(f32, f32, f32)` arm,
        // which is byte-identical to the pre-#750 kernel. The store dtype follows
        // the store leg of the triple, so `dst` is allocated and tagged to match.
        use crate::kernels::{GemvFloat, GemvTriple};
        let ggml_dtype: crate::kernels::GgmlDType = self.dtype.into();
        let triple = match storage.dtype() {
            DType::F32 => GemvTriple::F32F32F32,
            DType::F16 if ggml_dtype.has_gemv_triple() => GemvTriple::F16F32F16,
            // An f16 activation on a weight whose core has no f16 arm (dense/Q8),
            // or any other activation dtype, is not a triple this dispatch can
            // serve. Refuse rather than read the buffer under the wrong dtype
            // (§8.1.4: the kernel reads `src1` as the triple's activation leg, so
            // a mismatched buffer is silent corruption, not an error).
            other => crate::bail!(
                "quantized matmul-vec has no ({other:?}, f32, _) arm for weight {:?}",
                self.dtype
            ),
        };
        let store_dtype = match triple.store() {
            GemvFloat::F16 => DType::F16,
            GemvFloat::F32 => DType::F32,
        };

        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), store_dtype)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        // One launch over the `m` rows: the matmul-vec kernels take the
        // activation rows on the grid's second axis (`ne11`, stepped by
        // `nb11`), each row its own dot products — what a launch per row
        // computes, without the launch per row (a decode batch's `B` rows).
        crate::kernels::call_quantized_matmul_mv_t(
            device.device(),
            &encoder,
            device.kernels(),
            ggml_dtype,
            triple,
            (1, m, n, k),
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &self.buffer,
            0,
            &dst,
        )
        .map_err(MetalError::from)?;
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), store_dtype);
        Ok((dst_storage, dst_shape))
    }

    /// One f32 row through the matvec with `store` on its store
    /// (`kernels::call_quantized_matmul_mv_store`): a fresh `[n]` f32 under
    /// [`MvStore::Silu`](crate::kernels::MvStore::Silu); under `MulInto` the
    /// products multiplied into `into`'s `[n]` f32 elements, nothing returned.
    /// The caller has checked the kind, the row and the layouts.
    pub fn fwd_mv_store(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        store: crate::kernels::MvStore,
        into: Option<(&MetalStorage, &crate::Layout)>,
    ) -> Result<Option<MetalStorage>> {
        use crate::MetalError;
        let (n, k) = self_shape.dims2()?;
        if !layout.is_contiguous() || layout.shape().elem_count() != k || storage.dtype() != DType::F32 {
            crate::bail!("quantized matvec store block: one contiguous f32 row of {k}, got {layout:?}")
        }
        let device = storage.device().clone();
        let ggml_dtype: crate::kernels::GgmlDType = self.dtype.into();
        let encoder = device.command_encoder()?;
        let lhs_offset = layout.start_offset() * storage.dtype().size_in_bytes();
        match into {
            None => {
                let dst = device
                    .new_buffer_builder()
                    .with_size_for(n, DType::F32)
                    .with_label("qmatmul_store")
                    .build()?;
                crate::kernels::call_quantized_matmul_mv_store(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    ggml_dtype,
                    (n, k),
                    storage.buffer(),
                    lhs_offset,
                    &self.buffer,
                    0,
                    &dst,
                    store,
                )
                .map_err(MetalError::from)?;
                Ok(Some(crate::MetalStorage::new(dst, device.clone(), n, DType::F32)))
            }
            Some((dst, dst_layout)) => {
                if !dst_layout.is_contiguous()
                    || dst_layout.shape().elem_count() != n
                    || dst.dtype() != DType::F32
                {
                    crate::bail!("quantized matvec store block: the destination is not a contiguous f32 [{n}], {dst_layout:?}")
                }
                crate::kernels::call_quantized_matmul_mv_store(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    ggml_dtype,
                    (n, k),
                    storage.buffer(),
                    lhs_offset,
                    &self.buffer,
                    dst_layout.start_offset() * DType::F32.size_in_bytes(),
                    dst.buffer(),
                    store,
                )
                .map_err(MetalError::from)?;
                Ok(None)
            }
        }
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        // A handful of rows go through the matmul-vec kernel once per row:
        // the mm kernels tile the rows by 32 and dequantize whole tiles, so
        // at a decode batch's `m` (a few rows, `[B, 1, k]` or `[1, B, k]`)
        // they cost several times the per-row gemvs; and per-row gemvs make
        // every row exactly what it computes alone (the batched decode's
        // identity). Prefill widths keep the mm kernels.
        let rows_total = src_shape.elem_count() / k.max(1);
        if src_shape.dim(D::Minus2)? == 1 || rows_total <= Self::qmv_rows_max() {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        // The dequantise-once scratch is taken BEFORE the encoder guard
        // (`new_persistent_private` may open its own encoder; the commands
        // lock is not reentrant).
        let dequant_from = dequant_once_from();
        let (n_rows_w, k_w) = (self_shape.dims()[self_shape.rank() - 2], k);
        let dequant_scratch_buf = if dequant_from > 0
            && rows_total >= dequant_from
            && self_shape.rank() == 2
            && k_w.is_multiple_of(16)
            && dequant_once_kind(self.dtype.into())
        {
            Some(dequant_scratch(&device, n_rows_w * k_w * 2)?)
        } else {
            None
        };
        let encoder = device.command_encoder()?;

        // The GEMM arms read f32 activations; an f16 activation past the
        // matmul-vec's rows is refused here (#907's finding: it panicked on
        // an assert), not read under the wrong dtype.
        if storage.dtype() != DType::F32 {
            crate::bail!(
                "quantized matmul on Metal reads f32 activations at {rows_total} rows (the GEMM arms); got {:?} for weight {:?} — the f16 arm is the matmul-vec's, rows <= 8",
                storage.dtype(),
                self.dtype
            )
        }

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        // The stems at a prefill's rows: dequantised to half ONCE into a
        // persistent scratch, then the f16 GEMM arm reads plain halves —
        // measured 4.3 → 3.2 ms on `[10240 × 2560]` Q4_K at 512 rows, the
        // block unpack being the difference (P24/P26). The scratch is one
        // Private buffer a device, sized to the largest stem seen; the
        // encoder's hazard chain orders each stem's dequant after the
        // previous stem's GEMM read of it.
        if let Some(scratch) = dequant_scratch_buf {
            crate::kernels::call_quantized_dequant_rows_f16(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (n_rows_w, k_w),
                &self.buffer,
                &scratch,
            )
            .map_err(MetalError::from)?;
            // The steel gemm over the scratch (`LLOOM_QMM_DEQUANT_GEMM=steel`,
            // the A/B arm): the activations cast to half once (the quantized
            // mm converts them to half per tile in its threadgroup load —
            // the same operand precision), the result float from the kernel
            // (`OutT`), no cast back. A different tile loop, so a different
            // reduction order than the unpacking arm — and measured no faster
            // than the quantized f16 arm below (P26), which is the default.
            if dequant_gemm_steel() && layout.is_contiguous() {
                let x16 = device
                    .new_buffer_builder()
                    .with_size_for(rows_total * k, DType::F16)
                    .with_label("qmatmul_x16")
                    .build()?;
                crate::kernels::call_cast_contiguous(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    "cast_f32_f16",
                    DType::F32.size_in_bytes(),
                    rows_total * k,
                    crate::kernels::BufferOffset {
                        buffer: storage.buffer(),
                        offset_in_bytes: layout.start_offset() * DType::F32.size_in_bytes(),
                    },
                    &x16,
                )
                .map_err(MetalError::from)?;
                crate::kernels::call_mlx_gemm_out(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    crate::kernels::GemmDType::F16,
                    crate::kernels::GemmDType::F32,
                    (1, rows_total, n, k),
                    &[k, 1],
                    0,
                    &x16,
                    // The weight `[n, k]` read as `[k, n]`: transposed.
                    &[1, k],
                    0,
                    &scratch,
                    &dst,
                )
                .map_err(MetalError::from)?;
                drop(encoder);
                let dst_storage =
                    MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
                return Ok((dst_storage, dst_shape));
            }
            let src0_l = crate::Layout::contiguous(vec![1, 1, n_rows_w, k_w]);
            let src0_stride: Vec<usize> = src0_l
                .stride()
                .iter()
                .map(|x| x * DType::F16.size_in_bytes())
                .collect();
            let src1_l = crate::Layout::contiguous(
                [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
            );
            crate::kernels::call_quantized_matmul_mm_t(
                device.device(),
                &encoder,
                device.kernels(),
                crate::kernels::GgmlDType::F16,
                src0_l.dims(),
                &src0_stride,
                &scratch,
                0,
                src1_l.dims(),
                &src1_l
                    .stride()
                    .iter()
                    .map(|x| x * DType::F32.size_in_bytes())
                    .collect::<Vec<_>>(),
                storage.buffer(),
                src1_l.start_offset() * storage.dtype().size_in_bytes(),
                dst_shape.dims(),
                0,
                &dst,
            )
            .map_err(MetalError::from)?;
            let dst_storage =
                MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
            return Ok((dst_storage, dst_shape));
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        crate::kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            0,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Routed-expert matmul over a stacked quantized weight `[n_experts, n, k]`
    /// (the MoE seam `QTensor::indexed_moe_forward`, CUDA's contract): `input`
    /// `[batch, in_dim1, k]` f32 with `in_dim1 ∈ {1, topk}`, `ids` `[batch,
    /// topk]` u32 → `[batch, topk, n]` f32, one `kernel_mul_mv_id` dispatch
    /// or, on `arm` = [`super::MoeArm::Gemm`], the GEMM with the rows grouped
    /// by expert. The weight stays packed; each core dequantizes in-kernel
    /// (§3.3a).
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &Layout,
        ids: &MetalStorage,
        ids_l: &Layout,
        arm: super::MoeArm,
        act: DType,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        use crate::kernels::GemvTriple;
        let (n_experts, n, k) = self_shape.dims3()?;
        if !input_l.is_contiguous() || !ids_l.is_contiguous() {
            crate::bail!("indexed_moe_forward needs contiguous input and ids")
        }
        let ggml_dtype: crate::kernels::GgmlDType = self.dtype.into();
        let (triple, store_dtype) =
            self.routed_triple(input.dtype(), act, "indexed_moe_forward")?;
        if ids.dtype() != DType::U32 {
            crate::bail!("indexed_moe_forward ids must be u32, got {:?}", ids.dtype())
        }
        let (batch, in_dim1, k_in) = input_l.shape().dims3()?;
        let (batch_ids, topk) = ids_l.shape().dims2()?;
        if k_in != k {
            crate::bail!("indexed_moe_forward: input k {k_in} != weight k {k}")
        }
        if batch != batch_ids {
            crate::bail!("indexed_moe_forward: input batch {batch} != ids batch {batch_ids}")
        }
        if in_dim1 != 1 && in_dim1 != topk {
            crate::bail!("indexed_moe_forward: input dim1 {in_dim1} must be 1 or topk {topk}")
        }
        let device = self.device.clone();
        let dst_shape = Shape::from((batch, topk, n));
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), store_dtype)
            .with_label("qmatmul_id")
            .build()?;
        // The rows-grouped-by-expert GEMM for a tile (lloom #749 P22): the
        // caller's arm, f32 activations, every weight kind with a
        // `kernel_mul_mm_id_mapped` arm. The GEMM reads K in 32-wide blocks,
        // as the f16-weight GEMM does (#749 P30 (1)): a K off the block
        // reads past the row, so such a weight keeps the matmul-vec at every
        // row count.
        if arm == super::MoeArm::Gemm
            && triple == GemvTriple::F32F32F32
            && mm_id_has_arm(ggml_dtype)
            && crate::shape_req!(moe_mm_id, k % 32 == 0; k)
        {
            let bn = moe_mm_bn(batch * topk, n_experts);
            let bound = crate::kernels::mm_id_block_bound(batch, topk, n_experts, bn);
            let rowids = device
                .new_buffer_builder()
                .with_size_for(batch * topk, DType::U32)
                .with_label("moe_rowids")
                .build()?;
            let blocks = device
                .new_buffer_builder()
                .with_size_for(3 * bound, DType::U32)
                .with_label("moe_blocks")
                .build()?;
            let meta = device
                .new_buffer_builder()
                .with_size_for(1, DType::U32)
                .with_label("moe_rowmap_meta")
                .build()?;
            let encoder = device.command_encoder()?;
            crate::kernels::call_moe_rowmap(
                device.device(),
                &encoder,
                device.kernels(),
                batch,
                topk,
                n_experts,
                ids.buffer(),
                ids_l.start_offset() * DType::U32.size_in_bytes(),
                &rowids,
                &blocks,
                &meta,
                bound,
                bn,
            )
            .map_err(MetalError::from)?;
            crate::kernels::call_quantized_matmul_mm_id_mapped(
                device.device(),
                &encoder,
                device.kernels(),
                ggml_dtype,
                (n_experts, n, k),
                (batch, topk, in_dim1),
                &self.buffer,
                input.buffer(),
                input_l.start_offset() * input.dtype().size_in_bytes(),
                &rowids,
                &blocks,
                &meta,
                bound,
                bn,
                &dst,
            )
            .map_err(MetalError::from)?;
            let out = MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), store_dtype);
            return Ok((out, dst_shape));
        }
        let encoder = device.command_encoder()?;
        crate::kernels::call_quantized_matmul_mv_id_t(
            device.device(),
            &encoder,
            device.kernels(),
            ggml_dtype,
            triple,
            (n_experts, n, k),
            (batch, topk, in_dim1),
            &self.buffer,
            input.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        let out = MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), store_dtype);
        Ok((out, dst_shape))
    }

    /// Gate and up of one routed layer in ONE launch (`kernel_mul_mv_id2`):
    /// `other` is the second stacked weight of the SAME kind and shape; the
    /// result is `[2, batch, topk, n]` in the activation's triple's store
    /// dtype — plane set 0 this weight's, plane set 1 `other`'s, each what
    /// [`Self::indexed_moe_forward`]'s matmul-vec computes for it on the same
    /// activations — and `None`, laid out in its grid as `order` says. On
    /// [`super::PairOrder::Split`] the pair's kernel runs once a half into a
    /// buffer each: this weight's `[batch, topk, n]` and `Some` of `other`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn indexed_moe_forward_pair(
        &self,
        other: &Self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &Layout,
        ids: &MetalStorage,
        ids_l: &Layout,
        order: super::PairOrder,
        act: DType,
    ) -> Result<(MetalStorage, Option<MetalStorage>, Shape)> {
        use crate::MetalError;
        let (n_experts, n, k) = self_shape.dims3()?;
        if !input_l.is_contiguous() || !ids_l.is_contiguous() {
            crate::bail!("indexed_moe_forward_pair needs contiguous input and ids")
        }
        // Both weights hold `self_shape` of one kind. A pooled buffer may be
        // longer than the tensor in it, so each length is checked against
        // what the shape needs, never against the other's.
        let need = self_shape.elem_count() * self.dtype.type_size() / self.dtype.block_size();
        if other.dtype != self.dtype || other.buffer.length() < need || self.buffer.length() < need
        {
            crate::bail!(
                "indexed_moe_forward_pair: the two weights differ ({:?}/{} vs {:?}/{}, {} needed)",
                self.dtype,
                self.buffer.length(),
                other.dtype,
                other.buffer.length(),
                need
            )
        }
        let (triple, store_dtype) =
            self.routed_triple(input.dtype(), act, "indexed_moe_forward_pair")?;
        if ids.dtype() != DType::U32 {
            crate::bail!("indexed_moe_forward_pair ids must be u32, got {:?}", ids.dtype())
        }
        let ggml_dtype: crate::kernels::GgmlDType = self.dtype.into();
        let (batch, in_dim1, k_in) = input_l.shape().dims3()?;
        let (batch_ids, topk) = ids_l.shape().dims2()?;
        if k_in != k || batch != batch_ids || (in_dim1 != 1 && in_dim1 != topk) {
            crate::bail!(
                "indexed_moe_forward_pair: input {:?} / ids {:?} do not fit weight {:?}",
                input_l.shape(),
                ids_l.shape(),
                self_shape
            )
        }
        use super::PairOrder;
        use crate::kernels::PairGrid;
        let device = self.device.clone();
        let launch =
            |grid: PairGrid, a: &Buffer, b: &Buffer, shape: &Shape| -> Result<MetalStorage> {
                let dst = device
                    .new_buffer_builder()
                    .with_size_for(shape.elem_count(), store_dtype)
                    .with_label("qmatmul_id2")
                    .build()?;
                let encoder = device.command_encoder()?;
                crate::kernels::call_quantized_matmul_mv_id2_t(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    ggml_dtype,
                    triple,
                    grid,
                    (n_experts, n, k),
                    (batch, topk, in_dim1),
                    a,
                    b,
                    input.buffer(),
                    input_l.start_offset() * input.dtype().size_in_bytes(),
                    ids.buffer(),
                    ids_l.start_offset() * DType::U32.size_in_bytes(),
                    &dst,
                )
                .map_err(MetalError::from)?;
                Ok(MetalStorage::new(
                    dst,
                    device.clone(),
                    shape.elem_count(),
                    store_dtype,
                ))
            };
        let Some(grid) = order.grid() else {
            let shape = Shape::from((batch, topk, n));
            let g = launch(PairGrid::OneHalf, &self.buffer, &self.buffer, &shape)?;
            let u = launch(PairGrid::OneHalf, &other.buffer, &other.buffer, &shape)?;
            return Ok((g, Some(u), shape));
        };
        let shape = Shape::from((2, batch, topk, n));
        let both = launch(grid, &self.buffer, &other.buffer, &shape)?;
        Ok((both, None, shape))
    }

    /// The routed launches' `(activation, f32, store)` triple for an `input`
    /// activation computed in `act` on this weight, and the store's dtype:
    /// f16 reads and stores f16 on the quantized weight rows, f32 keeps the
    /// byte-identical `_f32` arm, and an f32 input computed in f16 is the f16
    /// arm with the rounding in the load (`F32AsF16F32F16`). Any other pairing
    /// is refused, never read under the wrong dtype (§8.1.4).
    fn routed_triple(
        &self,
        input: DType,
        act: DType,
        op: &str,
    ) -> Result<(crate::kernels::GemvTriple, DType)> {
        use crate::kernels::{GemvFloat, GemvTriple};
        let ggml_dtype: crate::kernels::GgmlDType = self.dtype.into();
        let f16_arm = ggml_dtype.has_gemv_triple();
        let triple = match (input, act) {
            (DType::F32, DType::F32) => GemvTriple::F32F32F32,
            (DType::F16, DType::F16) if f16_arm => GemvTriple::F16F32F16,
            (DType::F32, DType::F16) if f16_arm => GemvTriple::F32AsF16F32F16,
            (input, act) => crate::bail!(
                "{op} on Metal has no ({input:?} as {act:?}, f32, _) arm for weight {:?}",
                self.dtype
            ),
        };
        let store = match triple.store() {
            GemvFloat::F16 => DType::F16,
            GemvFloat::F32 => DType::F32,
        };
        Ok((triple, store))
    }

    /// Overwrite `len` bytes at `offset` with what `fill` writes: the bytes go
    /// into a Shared staging buffer on the host and are copied into this
    /// storage by a `copy2d` dispatch on the current compute encoder —
    /// deliberately NOT a blit, so the write sits in the encoder's hazard
    /// chain and is ordered after every earlier dispatch that read the
    /// destination (a blit encoder waits only on the destination's last
    /// writer).
    ///
    /// The staging buffer is unpooled (`MetalDevice::new_unpooled_shared`, one
    /// allocation per call, freed when the copy's handle drops): a pooled
    /// Shared buffer written by the host is the one tenant class whose reuse
    /// gate is known to leak at long context (lloom #748, `docs/invariants.md`
    /// §15.1m), and a leaked staging buffer would land a stale plane. The
    /// pooled path stays behind `LLOOM_STREAM_POOLED_STAGING` (read once) as
    /// the A/B instrument for whoever fixes the gate; never the default.
    pub fn write_with(
        &self,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let staging = if pooled_staging() {
            self.device.allocate_buffer(len)?
        } else {
            self.device.new_unpooled_shared(len)?
        };
        self.write_via(offset, len, staging, fill, None)
    }

    /// [`Self::write_with`] staged through `ring` — a buffer reused once the
    /// command buffer holding its last copy has completed, so a steady-state
    /// stream allocates nothing per load.
    pub fn write_with_ring(
        &self,
        offset: usize,
        len: usize,
        ring: &mut crate::kernels::metal::staging::StagingRing,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let (entry, staging) = ring.acquire(&self.device, len)?;
        self.write_via(offset, len, staging, fill, Some((ring, entry)))
    }

    /// [`Self::write_with_ring`] for bytes ALREADY in `staging` — a prefetch
    /// thread filled the ring buffer off the encoding thread
    /// (`nn::expert_prefetch`); this is the copy alone, on the current
    /// compute encoder, the entry stamped into `ring` after the encode as
    /// `write_with_ring` stamps it. The write is ordered after every earlier
    /// reader of the slot by the encoder's hazard chain exactly as the
    /// synchronous path's is: the thread that filled the bytes never touched
    /// the slot.
    pub fn write_staged(
        &self,
        offset: usize,
        len: usize,
        staging: BufferHandle,
        ring: &mut crate::kernels::metal::staging::StagingRing,
        entry: Option<usize>,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if staging.size() < len {
            crate::bail!(
                "write_staged: a {}-byte staging buffer for {len} bytes",
                staging.size()
            )
        }
        self.write_via(offset, len, staging, &mut |_| Ok(()), Some((ring, entry)))
    }

    fn write_via(
        &self,
        offset: usize,
        len: usize,
        staging: BufferHandle,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
        ring: Option<(
            &mut crate::kernels::metal::staging::StagingRing,
            Option<usize>,
        )>,
    ) -> Result<()> {
        use crate::MetalError;
        if offset + len > self.buffer.length() {
            crate::bail!(
                "write_with: {offset}+{len} past the {}-byte buffer",
                self.buffer.length()
            )
        }
        let ptr = std::hint::black_box(staging.contents());
        if ptr.is_null() {
            crate::bail!("write_with: the staging buffer has no CPU mapping (len={len})")
        }
        // SAFETY: a pooled Shared buffer of at least `len` bytes whose last
        // GPU reader has completed (pool reuse waits for the epoch), mapped at
        // `ptr`; the slice is dropped before the buffer is bound.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        fill(dst)?;
        // Four bytes per thread when the range allows it, one otherwise.
        let (kernel, width) = if offset.is_multiple_of(4) && len.is_multiple_of(4) {
            (crate::kernels::copy2d::U32, 4)
        } else {
            (crate::kernels::copy2d::U8, 1)
        };
        let n = len / width;
        let encoder = self.device.command_encoder()?;
        // The encoder guard holds the command stream: no cut can move the copy
        // into a later command buffer than the epoch stamped here.
        if let Some((ring, entry)) = ring {
            ring.stamp(entry, &staging, ring.clock().current_epoch());
        }
        crate::kernels::call_copy2d(
            self.device.device(),
            &encoder,
            self.device.kernels(),
            kernel,
            &staging,
            &self.buffer,
            1,
            n,
            n,
            n,
            0,
            offset,
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_data_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

/// `LLOOM_STREAM_POOLED_STAGING` (read once): stage streamed weight loads
/// through the Shared buffer pool instead of an unpooled buffer per load. An
/// A/B instrument for the pool's host-upload reuse gate (lloom #748); off.
fn pooled_staging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().pool.stream_pooled_staging)
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device
        .new_buffer_builder()
        .with_data(data)
        .with_label("qstorage_load_quantized")
        .build()?;
    // A weight upload is a Shared buffer (`new_buffer_with_data`): its pages
    // are the process's, compressible under pressure like the slot pools
    // (#749 P27); wired on request (`LLOOM_WIRE_WEIGHTS`).
    if crate::metal_backend::wire_weights_on() {
        device.wire_buffer(&buffer);
    }
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

/// # The `assert!` here does not survive optimisation (#454)
///
/// This is the same shape `metal_backend::read_to_vec` carried before #366,
/// in a second file: `contents()`, a bare `assert!`, then a `slice::from_raw_parts`.
/// §11.5d measured that assert absent from the release binary — its panic
/// string does not appear and the emitted code goes `contents()` → a *length*
/// test → `memcpy`, with no test on the pointer — and §14.5c measured
/// `panic!`-inside-an-`if` going the same way. #455 repaired the two sites in
/// `metal_backend`; this one was not in that survey's population.
///
/// Measured for #454 (`measurements/issue-454-raw/guard-probe/`, disassembly
/// in `measurements/issue-454-raw/shape-disassembly.txt`), reading the emitted
/// code rather than the source: in a bare-`assert!` function the pointer
/// returned by `objc_msgSend` goes straight to `ldrb` **with no `cbz` on it at
/// all**, at `opt-level=3` with LTO *off* as well as on — so this is not an
/// LTO-only hazard, and the tree compiles with `lto = "thin"` anyway (#445).
/// With `black_box` the same function stores the pointer, reloads it and
/// branches on `cbz`, in every profile tested.
///
/// The guard is load-bearing here because the buffer can be Private: a
/// `QMetalStorage`'s buffer comes from `new_buffer` →
/// `PRIVATE_RESOURCE_OPTIONS`, and §15.2 #12a records a Private allocation as
/// CPU-mapped below 16384 B and never at or above it. A quantized tensor's
/// storage is far above that threshold, so `contents()` returning null is the
/// expected case rather than a hypothetical one.
fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    // `black_box` makes the pointer opaque, so the test survives whatever the
    // optimiser can prove about the ObjC getter. Same repair as #445's, and it
    // costs one unfoldable move on a path that then copies `n` elements.
    let ptr = std::hint::black_box(buffer.contents()) as *const T;
    assert!(
        !ptr.is_null(),
        "quantized read_to_vec: buffer has no CPU mapping (len={}, n={n})",
        buffer.length()
    );
    // SAFETY: `ptr` is non-null and CPU-mapped, and the caller has waited for
    // any GPU work that writes it.
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for crate::kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => crate::kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => crate::kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => crate::kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => crate::kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => crate::kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => crate::kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => crate::kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => crate::kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => crate::kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => crate::kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => crate::kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => crate::kernels::GgmlDType::Q8K,
            GgmlDType::F16 => crate::kernels::GgmlDType::F16,
            GgmlDType::F32 => crate::kernels::GgmlDType::F32,
            GgmlDType::BF16 => crate::kernels::GgmlDType::BF16,
            GgmlDType::IQ2XXS => crate::kernels::GgmlDType::IQ2XXS,
            GgmlDType::IQ2XS => crate::kernels::GgmlDType::IQ2XS,
            GgmlDType::IQ2S => crate::kernels::GgmlDType::IQ2S,
            GgmlDType::IQ3XXS => crate::kernels::GgmlDType::IQ3XXS,
            GgmlDType::IQ3S => crate::kernels::GgmlDType::IQ3S,
            GgmlDType::IQ4NL => crate::kernels::GgmlDType::IQ4NL,
            GgmlDType::IQ4XS => crate::kernels::GgmlDType::IQ4XS,
            GgmlDType::Q2_0 => crate::kernels::GgmlDType::Q2_0,
        }
    }
}

/// Rows from which a tile's routed matmul takes the GEMM with the rows
/// grouped by expert (`kernel_mul_mm_id_mapped`) instead of the matmul-vec
/// ([`super::MoeArm::for_rows`]): `LLOOM_MOE_MM_FROM` (0 = never), read
/// once; default 64
/// ([`DEFAULT_MOE_MM_FROM`]) — a decode step (1 row, a batch of up to 8, an
/// MTP verify of K+1) keeps the byte-identical matmul-vec, a prefill tile
/// takes the GEMM. Settable for the tests and an A/B (`set_moe_mm_from`).
pub fn moe_mm_from() -> usize {
    let v = MOE_MM_FROM.load(std::sync::atomic::Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let from = crate::tuning::tuning().moe.mm_from;
    MOE_MM_FROM.store(from, std::sync::atomic::Ordering::Relaxed);
    from
}

pub const DEFAULT_MOE_MM_FROM: usize = 64;

/// Override [`moe_mm_from`] for the rest of the process.
pub fn set_moe_mm_from(from: usize) {
    MOE_MM_FROM.store(from, std::sync::atomic::Ordering::Relaxed);
}

static MOE_MM_FROM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// The routed GEMM's row block for a tile of `pairs` over `n_experts`:
/// 16 columns when the experts see few rows each (the 32-column tile ran
/// 32 columns for ~10 real ones at 512 experts, P22), 32 otherwise.
/// `LLOOM_MOE_MM_BN` = 16 | 32 pins it (0 = this rule); `set_moe_mm_bn`.
pub fn moe_mm_bn(pairs: usize, n_experts: usize) -> usize {
    let pinned = MOE_MM_BN.load(std::sync::atomic::Ordering::Relaxed);
    let pinned = if pinned == usize::MAX {
        let v = crate::tuning::tuning().moe.mm_bn;
        MOE_MM_BN.store(v, std::sync::atomic::Ordering::Relaxed);
        v
    } else {
        pinned
    };
    match pinned {
        16 | 32 => pinned,
        _ => {
            if pairs <= MOE_MM_BN16_ROWS_PER_EXPERT * n_experts.max(1) {
                16
            } else {
                32
            }
        }
    }
}

/// Rows an expert sees on average at or under which the 16-column block runs.
pub const MOE_MM_BN16_ROWS_PER_EXPERT: usize = 16;

/// Pin [`moe_mm_bn`] (16 | 32; 0 = the rule) for the rest of the process.
pub fn set_moe_mm_bn(bn: usize) {
    MOE_MM_BN.store(bn, std::sync::atomic::Ordering::Relaxed);
}

static MOE_MM_BN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Whether a weight kind has a `kernel_mul_mm_id_mapped` arm.
fn mm_id_has_arm(dtype: crate::kernels::GgmlDType) -> bool {
    dtype.kind().mul_mm_id_mapped
}

/// `xs [.., k]` (f32) · `w [n, k]`ᵀ (f16) → `[.., n]` f32 through the
/// quantized GEMM's f16 arm (`kernel_mul_mm_f16_f32`: the weights read as
/// half, the activations as float, the accumulate float) — no cast launch
/// on either side, where the classical route casts the activations to
/// half, multiplies, and casts the product back (three launches beside the
/// GEMM, and the activations rounded to half first). `QMatMul::TensorF16`'s
/// Metal route past the mixed gemv's rows (lloom #749 P23).
pub fn matmul_f16_weight_f32(xs: &crate::Tensor, w: &crate::Tensor) -> Result<crate::Tensor> {
    use crate::MetalError;
    let (n, k) = w.dims2()?;
    if w.dtype() != DType::F16 || xs.dtype() != DType::F32 {
        crate::bail!(
            "matmul_f16_weight_f32: w {:?}, xs {:?}",
            w.dtype(),
            xs.dtype()
        )
    }
    // The GEMM's f16 arm reads K in 32-wide blocks; a K off the block
    // reads past the row (#749 P30 (1)).
    crate::shape_ensure!(f16_weight_gemm, k % 32 == 0; k =>
        "matmul_f16_weight_f32: K {k} is not a multiple of the 32-wide block");
    let xs = xs.contiguous()?;
    let w = w.contiguous()?;
    let (ws, wl) = w.storage_and_layout();
    let (xsto, xl) = xs.storage_and_layout();
    let (wm, xm) = match (&*ws, &*xsto) {
        (crate::Storage::Metal(a), crate::Storage::Metal(b)) => (a, b),
        _ => crate::bail!("matmul_f16_weight_f32: Metal tensors only"),
    };
    let x_dims = xl.dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != k || x_dims.len() > 4 {
        crate::bail!(
            "matmul_f16_weight_f32: xs {:?} against w [{n}, {k}]",
            x_dims
        )
    }
    let device = xm.device().clone();
    let mut out_dims = x_dims.to_vec();
    *out_dims.last_mut().expect("rank >= 1") = n;
    let dst_shape = Shape::from(out_dims.clone());
    let dst = device
        .new_buffer_builder()
        .with_size_for(dst_shape.elem_count(), DType::F32)
        .with_label("qmatmul_f16w")
        .build()?;
    let src0_l = crate::Layout::contiguous(vec![1, 1, n, k]);
    let src0_stride: Vec<usize> = src0_l
        .stride()
        .iter()
        .map(|x| x * DType::F16.size_in_bytes())
        .collect();
    let src1_l = crate::Layout::contiguous([vec![1; 4 - x_dims.len()], x_dims.to_vec()].concat());
    let dst_l = crate::Layout::contiguous([vec![1; 4 - out_dims.len()], out_dims.clone()].concat());
    let encoder = device.command_encoder()?;
    crate::kernels::call_quantized_matmul_mm_t(
        device.device(),
        &encoder,
        device.kernels(),
        crate::kernels::GgmlDType::F16,
        src0_l.dims(),
        &src0_stride,
        wm.buffer(),
        wl.start_offset() * DType::F16.size_in_bytes(),
        src1_l.dims(),
        &src1_l
            .stride()
            .iter()
            .map(|x| x * DType::F32.size_in_bytes())
            .collect::<Vec<_>>(),
        xm.buffer(),
        xl.start_offset() * DType::F32.size_in_bytes(),
        dst_l.dims(),
        0,
        &dst,
    )
    .map_err(MetalError::from)?;
    drop(encoder);
    let storage = MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
    Ok(crate::tensor::from_storage(
        crate::Storage::Metal(storage),
        dst_shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// Whether `QMatMul::TensorF16` takes [`matmul_f16_weight_f32`] on Metal
/// past the mixed gemv's rows: `LLOOM_QMM_F16_MM` (0 = the classical
/// cast-matmul-cast route), read once.
pub fn f16_weight_mm_route() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qmm.f16_mm)
}

/// Rows from which a quantized stem's Metal GEMM dequantises the weight to
/// half once and runs the f16 arm: `LLOOM_QMM_DEQUANT_FROM` (0 = never),
/// read once; default 64 — the prefill tiles; a decode step and the gate
/// prompts keep the in-kernel unpack. `set_dequant_once_from` for tests.
pub fn dequant_once_from() -> usize {
    let v = DEQUANT_ONCE_FROM.load(std::sync::atomic::Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let from = crate::tuning::tuning().qmm.dequant_from;
    DEQUANT_ONCE_FROM.store(from, std::sync::atomic::Ordering::Relaxed);
    from
}

pub const DEFAULT_DEQUANT_ONCE_FROM: usize = 64;

static DEQUANT_GEMM_STEEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);

/// Which GEMM reads the dequantised scratch: the quantized GEMM's f16 arm
/// (`LLOOM_QMM_DEQUANT_GEMM=qmm`, the default — the same tile loop as the
/// unpacking arm, so bit-identical to it) or the steel gemm with half
/// operands and a float result (`=steel`). Measured the same to the
/// millisecond on the stems (3.58 vs 3.58 ms on `[10240 × 2560]` at 512
/// rows; GPU busy 4.22–4.25 s either way over the 937-token prompt, issue
/// #749 P26): the steel kernel's 0.3 ms over the quantized arm on a
/// pre-cast half input is spent on the cast and the float store. The A/B
/// arm, kept. Read once; `set_dequant_gemm_steel` for tests.
pub fn dequant_gemm_steel() -> bool {
    let v = DEQUANT_GEMM_STEEL.load(std::sync::atomic::Ordering::Relaxed);
    if v != u8::MAX {
        return v != 0;
    }
    let steel = match crate::tuning::tuning().qmm.dequant_gemm.as_str() {
        "qmm" => false,
        "steel" => true,
        _ => DEFAULT_DEQUANT_GEMM_STEEL,
    };
    DEQUANT_GEMM_STEEL.store(steel as u8, std::sync::atomic::Ordering::Relaxed);
    steel
}

pub fn set_dequant_gemm_steel(on: bool) {
    DEQUANT_GEMM_STEEL.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

pub const DEFAULT_DEQUANT_GEMM_STEEL: bool = false;

pub fn set_dequant_once_from(from: usize) {
    DEQUANT_ONCE_FROM.store(from, std::sync::atomic::Ordering::Relaxed);
}

static DEQUANT_ONCE_FROM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// The weight kinds with a `kernel_dequant_rows_f16` arm (every quantized
/// kind; the float kinds have nothing to dequantise).
fn dequant_once_kind(dtype: crate::kernels::GgmlDType) -> bool {
    dtype.kind().dequant_rows_f16
}

/// One Private scratch a device for the dequantised stems, grown to the
/// largest asked for (outside the pools: a tenant with a fixed identity,
/// the hazard chain orders every stem's write after the last stem's read).
fn dequant_scratch(device: &MetalDevice, bytes: usize) -> Result<BufferHandle> {
    static SCRATCH: std::sync::OnceLock<
        parking_lot::Mutex<std::collections::HashMap<crate::metal_backend::DeviceId, BufferHandle>>,
    > = std::sync::OnceLock::new();
    let map = SCRATCH.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
    let key = device.id();
    let mut g = map.lock();
    if let Some(b) = g.get(&key) {
        if b.size() >= bytes {
            return Ok(BufferHandle::clone(b));
        }
    }
    let b = device.new_persistent_private(bytes)?;
    g.insert(key, BufferHandle::clone(&b));
    Ok(b)
}

impl super::PairOrder {
    /// The one launch's grid this order dispatches, or `None` for
    /// [`super::PairOrder::Split`], which launches the pair's kernel once a
    /// half on [`crate::kernels::PairGrid::OneHalf`].
    pub fn grid(self) -> Option<crate::kernels::PairGrid> {
        use super::PairOrder;
        use crate::kernels::PairGrid;
        match self {
            PairOrder::Half => Some(PairGrid::HalfSlowest),
            PairOrder::HalfCmp => Some(PairGrid::HalfSlowestCmp),
            PairOrder::Expert => Some(PairGrid::HalfByPlane),
            PairOrder::Tile => Some(PairGrid::HalfFastest),
            PairOrder::Split => None,
        }
    }
}

#[cfg(test)]
mod kind_table_tests {
    use super::GgmlDType;

    /// The kernel side's table (`kernels::QuantKind`, #892 B2) carries each
    /// format's block geometry; it must agree with this module's block
    /// definitions, the bytes a tensor is sized by.
    #[test]
    fn the_kernel_table_agrees_with_the_block_geometry() {
        const ALL: [GgmlDType; 23] = [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q8_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
            GgmlDType::F16,
            GgmlDType::F32,
            GgmlDType::BF16,
            GgmlDType::IQ2XXS,
            GgmlDType::IQ2XS,
            GgmlDType::IQ2S,
            GgmlDType::IQ3XXS,
            GgmlDType::IQ3S,
            GgmlDType::IQ4NL,
            GgmlDType::IQ4XS,
            GgmlDType::Q2_0,
        ];
        for d in ALL {
            let k: crate::kernels::GgmlDType = d.into();
            let kind = k.kind();
            assert_eq!(kind.block_size, d.block_size(), "{d:?} block_size");
            assert_eq!(kind.type_size, d.type_size(), "{d:?} type_size");
        }
    }
}
