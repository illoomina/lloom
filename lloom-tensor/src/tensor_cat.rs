use crate::{shape::Dim, Context, Error, Result, Shape, Tensor};

impl Tensor {
    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// # use lloom_tensor::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1)?;
    /// assert_eq!(c.shape().dims(), &[2, 6]);
    /// # Ok::<(), lloom_tensor::Error>(())
    /// ```
    pub fn cat<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let dim = dim.to_index(arg0.shape(), "cat")?;
        for arg in args {
            arg.as_ref().check_dim(dim, "cat")?;
        }
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg0.rank() != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: arg0.rank(),
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let all_copy2d_compat = args.iter().all(|v| {
            let t = v.as_ref();
            t.is_contiguous() || t.layout().outer_stride_for_dim(dim).is_some()
        });
        if all_copy2d_compat {
            Self::cat_contiguous(args, dim)
        } else if dim == 0 {
            Self::cat0(args)
        } else {
            let args: Vec<Tensor> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<Tensor>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
            let next_offset = offsets.last().context("empty offsets")? + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = crate::op::BackpropOp::new(args, |args| crate::op::Op::Cat(args, 0));
        let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage()
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }

    fn cat_contiguous<A: AsRef<Tensor>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[dim] = 0;
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == dim {
                    cat_dims[dim] += v2;
                }
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let cat_target_dim_len = cat_dims[dim];
        let block_size: usize = cat_dims.iter().skip(1 + dim).product();
        let shape = Shape::from(cat_dims);
        let op = crate::op::BackpropOp::new(args, |args| crate::op::Op::Cat(args, dim));
        let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        let mut dst_o = 0;
        for arg in args.iter() {
            let arg = arg.as_ref();
            let arg_dims = arg.shape().dims();
            let d1: usize = arg_dims.iter().take(dim).product();
            let d2 = block_size * arg_dims[dim];
            let dst_s = block_size * cat_target_dim_len;
            let src_o = arg.layout().start_offset();
            let src_s = if arg.is_contiguous() {
                d2
            } else {
                arg.layout().outer_stride_for_dim(dim).unwrap_or(d2)
            };
            arg.storage()
                .copy2d(&mut storage, d1, d2, src_s, dst_s, src_o, dst_o)?;
            dst_o += d2;
        }
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }

    /// Set the values on `self` using values from `src`. The copy starts at the specified
    /// `offset` for the target dimension `dim` on `self`.
    /// `self` and `src` must have the same shape except on dimension `dim` where the `self` size
    /// has to be greater than or equal to `offset` plus the `src` size.
    ///
    /// Note that this modifies `self` in place and as such is not compatible with
    /// back-propagation.
    pub fn slice_set<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if !self.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set" }.bt())?
        }
        // A strided source is written where it lies, through the backend's
        // strided-to-strided copy, so a `transpose` view reaches the window
        // in one dispatch. A backend without one materialises the source
        // first, which is what every caller did before.
        if !src.is_contiguous() {
            self.slice_set_checks(src, dim, offset)?;
            let dst_l = self.layout().narrow(dim, offset, src.dims()[dim])?;
            let done =
                src.storage()
                    .copy_strided_to(&mut self.storage_mut(), src.layout(), &dst_l)?;
            if done {
                return Ok(());
            }
            return self.slice_set(&src.contiguous()?, dim, offset);
        }
        self.slice_set_checks(src, dim, offset)?;
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        src.storage().copy2d(
            &mut self.storage_mut(),
            d1,
            d2,
            /* src_s */ d2,
            /* dst_s */ block_size * self.dims()[dim],
            src_o,
            dst_o,
        )?;

        Ok(())
    }

    /// The shape, dtype, device and storage checks `slice_set` makes before
    /// writing, shared by its contiguous and strided source paths.
    fn slice_set_checks(&self, src: &Self, dim: usize, offset: usize) -> Result<()> {
        if self.same_storage(src) {
            crate::bail!("cannot use slice_set when self and src share their storage")
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.device().location() != src.device().location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                crate::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                crate::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        Ok(())
    }

    /// Write this `[n_seqs, n_kv_heads, seq_len, head_dim]` tensor's rows into
    /// `dst` at the token slots `slots` names, in place.
    ///
    /// `slots` is a `U32` tensor of `n_seqs * seq_len` entries: row `b * seq_len
    /// + t` is the token slot of sequence `b`'s `t`-th token. `dst` is a
    /// contiguous 4-D `[_, n_kv_heads, tokens, head_dim]` buffer reached as
    /// `b * dst_seq_stride + h * head_stride + slot * token_stride + d`, where
    /// `dst_seq_stride` is 0 for a page pool shared by every sequence
    /// (`DESIGN.md` §10.3c) and a batch stride for a `[B, H_kv, capacity, D]`
    /// cache. `slice_set` with a per-token destination — the KV append a page
    /// table needs, which a single `narrow` cannot express (§10.3e).
    ///
    /// `max_slot` is the largest entry of `slots`, from the host copy the
    /// caller built it from. It is taken as an argument rather than read back
    /// because reading a device tensor waits for every in-flight command
    /// buffer — measured at **+6 ms per token** on a path that costs 32 — and
    /// the slots are always computed on the host first. The bound is checked
    /// against the destination here, so a pool sized for fewer tokens than the
    /// slots name is an error rather than the silent overrun §3.5 describes.
    pub fn kv_scatter_into(
        &self,
        dst: &Self,
        slots: &Self,
        max_slot: usize,
        dst_seq_stride: usize,
        dst_head_stride: usize,
        dst_token_stride: usize,
    ) -> Result<()> {
        let (n_seqs, n_kv_heads, seq_len, head_dim) = self.dims4()?;
        if !self.is_contiguous() || !dst.is_contiguous() || !slots.is_contiguous() {
            Err(Error::RequiresContiguous { op: "kv-scatter" }.bt())?
        }
        if self.same_storage(dst) {
            crate::bail!("cannot use kv_scatter_into when self and dst share their storage")
        }
        if self.dtype() != dst.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: dst.dtype(),
                op: "kv-scatter",
            }
            .bt())?
        }
        if slots.dtype() != crate::DType::U32 {
            crate::bail!("kv_scatter_into needs U32 slots, got {:?}", slots.dtype())
        }
        if slots.elem_count() != n_seqs * seq_len {
            crate::bail!(
                "kv_scatter_into: {} slots for {n_seqs} x {seq_len} rows",
                slots.elem_count()
            )
        }
        let dst_dims = dst.dims4()?;
        if dst_dims.1 != n_kv_heads || dst_dims.3 != head_dim {
            crate::bail!(
                "kv_scatter_into: source [{n_seqs}, {n_kv_heads}, {seq_len}, {head_dim}] does \
                 not match destination {:?} on heads or head_dim",
                dst.dims()
            )
        }
        // Every slot must land inside the destination: the write is a kernel
        // under HazardTrackingModeUntracked and an overrun is silent (§3.5).
        let tokens = dst_dims.2;
        let seqs_reach = if n_seqs > 1 {
            (n_seqs - 1) * dst_seq_stride
        } else {
            0
        };
        let last = seqs_reach
            + (n_kv_heads - 1) * dst_head_stride
            + max_slot * dst_token_stride
            + head_dim;
        if max_slot >= tokens || last > dst.elem_count() {
            crate::bail!(
                "kv_scatter_into: slot {max_slot} reaches element {last} of a {}-element \
                 destination with {tokens} tokens",
                dst.elem_count()
            )
        }
        let geometry = crate::KvScatterGeometry {
            n_seqs,
            n_kv_heads,
            seq_len,
            head_dim,
            src_stride: [
                self.stride()[0],
                self.stride()[1],
                self.stride()[2],
                self.stride()[3],
            ],
            src_offset: self.layout().start_offset(),
            dst_seq_stride,
            dst_head_stride,
            dst_token_stride,
            dst_offset: dst.layout().start_offset(),
            slots_offset: slots.layout().start_offset(),
        };
        self.storage()
            .kv_scatter(&mut dst.storage_mut(), &slots.storage(), &geometry)
    }
}
