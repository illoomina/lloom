//! The command list a planned pass encodes, and how its fields follow the
//! step (`design/docs/planned-step-design.md` §5, §6.2).

use std::ops::Range;

use crate::kernels::metal::{Buffer, ComputePipeline, Grid, Size3};
use crate::kernels::MetalKernelError;

/// A scalar the host's state advance produces once per step: a position, a
/// cache length, a ring slot. Its value lives in [`StepValues`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepParam(pub u16);

/// One step's parameter values, indexed by [`StepParam`]. Reused step to step:
/// the advance overwrites it in place.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepValues {
    values: Vec<i64>,
}

impl StepValues {
    pub fn new(params: usize) -> Self {
        Self {
            values: vec![0; params],
        }
    }

    pub fn set(&mut self, param: StepParam, value: i64) {
        self.values[param.0 as usize] = value;
    }

    pub fn get(&self, param: StepParam) -> i64 {
        self.values[param.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// `base + Σ coefficient · parameter`: a field as a function of the step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Affine {
    base: i64,
    terms: Vec<(StepParam, i64)>,
}

impl Affine {
    pub fn constant(base: i64) -> Self {
        Self {
            base,
            terms: Vec::new(),
        }
    }

    pub fn new(base: i64, terms: &[(StepParam, i64)]) -> Self {
        Self {
            base,
            terms: terms.to_vec(),
        }
    }

    pub fn base(&self) -> i64 {
        self.base
    }

    pub fn terms(&self) -> &[(StepParam, i64)] {
        &self.terms
    }

    /// The field's value at `values`, or `None` when the arithmetic overflows.
    pub fn eval(&self, values: &StepValues) -> Option<i64> {
        let mut acc = self.base;
        for &(param, coefficient) in &self.terms {
            acc = acc.checked_add(coefficient.checked_mul(values.get(param))?)?;
        }
        Some(acc)
    }
}

/// The width of an inline integer field a patch writes, in the kernel's
/// argument layout. `usize` arguments are `U64`: they lower as `size_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    U32,
    I32,
    U64,
    I64,
}

impl Width {
    pub fn bytes(self) -> usize {
        match self {
            Width::U32 | Width::I32 => 4,
            Width::U64 | Width::I64 => 8,
        }
    }

    /// Write `value` into `out` in this width, native-endian. `false` when the
    /// value does not fit, which leaves `out` untouched.
    fn write(self, value: i64, out: &mut [u8]) -> bool {
        match self {
            Width::U32 => u32::try_from(value).map(|v| out.copy_from_slice(&v.to_ne_bytes())),
            Width::I32 => i32::try_from(value).map(|v| out.copy_from_slice(&v.to_ne_bytes())),
            Width::U64 => u64::try_from(value).map(|v| out.copy_from_slice(&v.to_ne_bytes())),
            Width::I64 => {
                out.copy_from_slice(&value.to_ne_bytes());
                Ok(())
            }
        }
        .is_ok()
    }
}

/// A buffer a list binds, by its place in a [`BindTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferRef(pub u32);

/// The buffers a list binds. The list names them by [`BufferRef`], so a
/// replaced session buffer or a grown base is one `set`, not a new list.
#[derive(Default)]
pub struct BindTable {
    buffers: Vec<Option<Buffer>>,
}

impl BindTable {
    pub fn with_len(len: usize) -> Self {
        Self {
            buffers: (0..len).map(|_| None).collect(),
        }
    }

    /// Bind `buffer` at `at`, returning what was there.
    pub fn set(&mut self, at: BufferRef, buffer: Buffer) -> Option<Buffer> {
        self.buffers[at.0 as usize].replace(buffer)
    }

    pub fn get(&self, at: BufferRef) -> Option<&Buffer> {
        self.buffers.get(at.0 as usize).and_then(Option::as_ref)
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// One argument of a command, in the order the kernel's `call_*` sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    /// `len` inline bytes at `index`, from the list's byte store at `start`.
    Bytes {
        index: u32,
        start: u32,
        len: u32,
    },
    /// A buffer at `index`, `offset` bytes into the handle the table holds.
    Buffer {
        index: u32,
        buffer: BufferRef,
        offset: usize,
        write: bool,
    },
    ThreadgroupMemory {
        index: u32,
        len: u32,
    },
}

/// An argument of a list, by position, for attaching a [`Patch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgId(u32);

/// What a patch overwrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchTarget {
    /// The buffer argument's offset: the patch's value replaces it.
    Offset,
    /// The integer field of `width` at byte `at` of a bytes argument.
    Bytes { at: u32, width: Width },
}

/// A field that follows the step: its argument, which part, and its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patch {
    pub(super) arg: u32,
    target: PatchTarget,
    value: Affine,
}

impl Patch {
    pub fn target(&self) -> PatchTarget {
        self.target
    }

    pub fn value(&self) -> &Affine {
        &self.value
    }
}

/// One dispatch: a pipeline, its arguments, its patches and its grid.
/// `tag` is the caller's name for the node it lowers.
#[derive(Clone)]
pub struct Command {
    pipeline: ComputePipeline,
    pub(super) args: Range<u32>,
    patches: Range<u32>,
    grid: Grid,
    threads: Size3,
    tag: u32,
}

impl Command {
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    pub fn threads(&self) -> Size3 {
        self.threads
    }

    pub fn tag(&self) -> u32 {
        self.tag
    }
}

/// A planned pass's commands, in encoding order.
#[derive(Clone, Default)]
pub struct CommandList {
    commands: Vec<Command>,
    args: Vec<Arg>,
    patches: Vec<Patch>,
    bytes: Vec<u8>,
    params: usize,
    max_bytes: usize,
    buffers: usize,
}

impl CommandList {
    /// An empty list whose patches may read `params` step parameters.
    pub fn new(params: usize) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    /// Begin a command on `pipeline`. Its arguments follow in the order the
    /// kernel sets them; [`CommandBuilder::dispatch_threadgroups`] or
    /// [`CommandBuilder::dispatch_threads`] ends it.
    pub fn command(&mut self, pipeline: ComputePipeline, tag: u32) -> CommandBuilder<'_> {
        CommandBuilder {
            args_start: self.args.len(),
            patches_start: self.patches.len(),
            bytes_start: self.bytes.len(),
            buffers_start: self.buffers,
            max_bytes_start: self.max_bytes,
            list: self,
            pipeline: Some(pipeline),
            tag,
        }
    }

    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    pub fn args(&self, command: &Command) -> &[Arg] {
        &self.args[command.args.start as usize..command.args.end as usize]
    }

    pub fn patches(&self, command: &Command) -> &[Patch] {
        &self.patches[command.patches.start as usize..command.patches.end as usize]
    }

    /// The bytes a [`Arg::Bytes`] holds.
    pub fn bytes(&self, start: u32, len: u32) -> &[u8] {
        &self.bytes[start as usize..(start + len) as usize]
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// How many step parameters the patches may read.
    pub fn params(&self) -> usize {
        self.params
    }

    /// One past the largest [`BufferRef`] the list binds: the table length it
    /// needs.
    pub fn buffers(&self) -> usize {
        self.buffers
    }

    /// The largest bytes argument, which bounds the encoder's patch scratch.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Drop every command from `len` on, with their arguments, patches and
    /// bytes: a node refused after its commands were written leaves the list
    /// as it was before the node.
    pub(super) fn truncate(&mut self, len: usize) {
        let Some(first) = self.commands.get(len) else {
            return;
        };
        let (args, patches) = (first.args.start as usize, first.patches.start as usize);
        let bytes = self.args[args..]
            .iter()
            .filter_map(|a| match *a {
                Arg::Bytes { start, .. } => Some(start as usize),
                _ => None,
            })
            .min()
            .unwrap_or(self.bytes.len());
        self.commands.truncate(len);
        self.args.truncate(args);
        self.patches.truncate(patches);
        self.bytes.truncate(bytes);
        self.buffers = self
            .args
            .iter()
            .filter_map(|a| match *a {
                Arg::Buffer { buffer, .. } => Some(buffer.0 as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        self.max_bytes = self
            .args
            .iter()
            .filter_map(|a| match *a {
                Arg::Bytes { len, .. } => Some(len as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);
    }

    /// Every buffer the list binds, once each, in first-bind order: the
    /// residency a step of this list needs.
    pub fn bound(&self) -> Vec<BufferRef> {
        let mut seen = vec![false; self.buffers];
        let mut out = Vec::new();
        for arg in &self.args {
            if let Arg::Buffer { buffer, .. } = *arg {
                if !std::mem::replace(&mut seen[buffer.0 as usize], true) {
                    out.push(buffer);
                }
            }
        }
        out
    }
}

fn invalid(what: String) -> MetalKernelError {
    MetalKernelError::InvalidInput(format!("command list: {what}"))
}

/// One command under construction. Dropped without a dispatch, it leaves the
/// list as it found it.
pub struct CommandBuilder<'a> {
    list: &'a mut CommandList,
    pipeline: Option<ComputePipeline>,
    tag: u32,
    args_start: usize,
    patches_start: usize,
    bytes_start: usize,
    buffers_start: usize,
    max_bytes_start: usize,
}

impl CommandBuilder<'_> {
    fn push(&mut self, arg: Arg) -> ArgId {
        let id = ArgId(self.list.args.len() as u32);
        self.list.args.push(arg);
        id
    }

    /// Inline bytes at argument `index`.
    pub fn bytes(&mut self, index: usize, bytes: &[u8]) -> ArgId {
        let start = self.list.bytes.len() as u32;
        self.list.bytes.extend_from_slice(bytes);
        self.list.max_bytes = self.list.max_bytes.max(bytes.len());
        self.push(Arg::Bytes {
            index: index as u32,
            start,
            len: bytes.len() as u32,
        })
    }

    /// A `usize` scalar, as `set_params!` binds one (`size_t`, 8 bytes).
    pub fn usize(&mut self, index: usize, value: usize) -> ArgId {
        self.bytes(index, &(value as u64).to_ne_bytes())
    }

    pub fn u32(&mut self, index: usize, value: u32) -> ArgId {
        self.bytes(index, &value.to_ne_bytes())
    }

    pub fn i64(&mut self, index: usize, value: i64) -> ArgId {
        self.bytes(index, &value.to_ne_bytes())
    }

    pub fn f32(&mut self, index: usize, value: f32) -> ArgId {
        self.bytes(index, &value.to_ne_bytes())
    }

    /// The buffer the table holds at `buffer`, `offset` bytes in.
    pub fn buffer(&mut self, index: usize, buffer: BufferRef, offset: usize, write: bool) -> ArgId {
        self.list.buffers = self.list.buffers.max(buffer.0 as usize + 1);
        self.push(Arg::Buffer {
            index: index as u32,
            buffer,
            offset,
            write,
        })
    }

    pub fn threadgroup_memory(&mut self, index: usize, len: usize) -> ArgId {
        self.push(Arg::ThreadgroupMemory {
            index: index as u32,
            len: len as u32,
        })
    }

    fn own_arg(&self, arg: ArgId) -> Result<Arg, MetalKernelError> {
        let at = arg.0 as usize;
        if at < self.args_start || at >= self.list.args.len() {
            return Err(invalid(format!(
                "patch on argument {at}, which is not this command's"
            )));
        }
        Ok(self.list.args[at])
    }

    fn check_params(&self, value: &Affine) -> Result<(), MetalKernelError> {
        match value
            .terms
            .iter()
            .find(|(p, _)| p.0 as usize >= self.list.params)
        {
            Some((p, _)) => Err(invalid(format!(
                "patch reads parameter {} of {}",
                p.0, self.list.params
            ))),
            None => Ok(()),
        }
    }

    /// `arg`'s offset follows `value`.
    pub fn patch_offset(&mut self, arg: ArgId, value: Affine) -> Result<(), MetalKernelError> {
        if !matches!(self.own_arg(arg)?, Arg::Buffer { .. }) {
            return Err(invalid("an offset patch on a non-buffer argument".into()));
        }
        self.check_params(&value)?;
        self.list.patches.push(Patch {
            arg: arg.0,
            target: PatchTarget::Offset,
            value,
        });
        Ok(())
    }

    /// The `width` field at byte `at` of bytes argument `arg` follows `value`.
    pub fn patch_bytes(
        &mut self,
        arg: ArgId,
        at: usize,
        width: Width,
        value: Affine,
    ) -> Result<(), MetalKernelError> {
        let Arg::Bytes { len, .. } = self.own_arg(arg)? else {
            return Err(invalid("a bytes patch on a non-bytes argument".into()));
        };
        if at
            .checked_add(width.bytes())
            .is_none_or(|end| end > len as usize)
        {
            return Err(invalid(format!(
                "a {}-byte field at {at} of a {len}-byte argument",
                width.bytes()
            )));
        }
        self.check_params(&value)?;
        self.list.patches.push(Patch {
            arg: arg.0,
            target: PatchTarget::Bytes {
                at: at as u32,
                width,
            },
            value,
        });
        Ok(())
    }

    fn finish(mut self, grid: Grid, threads: Size3) {
        // The encoder walks a command's arguments and patches together, so
        // patches are kept in argument order.
        self.list.patches[self.patches_start..].sort_by_key(|p| p.arg);
        let command = Command {
            pipeline: self.pipeline.take().expect("a builder finishes once"),
            args: self.args_start as u32..self.list.args.len() as u32,
            patches: self.patches_start as u32..self.list.patches.len() as u32,
            grid,
            threads,
            tag: self.tag,
        };
        self.list.commands.push(command);
    }

    pub fn dispatch_threadgroups(self, threadgroups: Size3, threads: Size3) {
        self.finish(Grid::Threadgroups(threadgroups), threads);
    }

    pub fn dispatch_threads(self, grid: Size3, threads: Size3) {
        self.finish(Grid::Threads(grid), threads);
    }
}

impl Drop for CommandBuilder<'_> {
    fn drop(&mut self) {
        if self.pipeline.is_some() {
            self.list.args.truncate(self.args_start);
            self.list.patches.truncate(self.patches_start);
            self.list.bytes.truncate(self.bytes_start);
            self.list.buffers = self.buffers_start;
            self.list.max_bytes = self.max_bytes_start;
        }
    }
}

/// Write `patch`'s value at `values` into `bytes`, the scratch copy of its
/// bytes argument.
pub(super) fn apply_bytes(
    patch: &Patch,
    values: &StepValues,
    bytes: &mut [u8],
) -> Result<(), MetalKernelError> {
    let PatchTarget::Bytes { at, width } = patch.target else {
        unreachable!("an offset patch on a bytes argument is refused at build");
    };
    let value = patch
        .value
        .eval(values)
        .ok_or_else(|| invalid("a bytes patch overflowed".into()))?;
    let field = &mut bytes[at as usize..at as usize + width.bytes()];
    if !width.write(value, field) {
        return Err(invalid(format!("{value} does not fit a {width:?} field")));
    }
    Ok(())
}

/// `patch`'s offset at `values`.
pub(super) fn offset_at(patch: &Patch, values: &StepValues) -> Result<usize, MetalKernelError> {
    patch
        .value
        .eval(values)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("an offset patch is negative or overflowed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const P0: StepParam = StepParam(0);
    const P1: StepParam = StepParam(1);

    #[test]
    fn an_affine_field_is_its_base_plus_each_term() {
        let mut v = StepValues::new(2);
        v.set(P0, 7);
        v.set(P1, -3);
        let f = Affine::new(1792, &[(P0, -256), (P1, 4)]);
        assert_eq!(f.eval(&v), Some(1792 - 7 * 256 - 12));
        assert_eq!(Affine::constant(5).eval(&v), Some(5));
        v.set(P0, i64::MAX);
        assert_eq!(f.eval(&v), None, "overflow is reported, not wrapped");
    }

    #[test]
    fn a_width_refuses_a_value_it_cannot_hold_and_writes_the_rest_exactly() {
        let mut out = [0xAAu8; 4];
        assert!(!Width::U32.write(-1, &mut out));
        assert_eq!(out, [0xAA; 4], "a refused write leaves the field");
        assert!(!Width::U32.write(1 << 32, &mut out));
        assert!(!Width::I32.write(i64::from(i32::MAX) + 1, &mut out));
        assert!(Width::I32.write(-2, &mut out));
        assert_eq!(out, (-2i32).to_ne_bytes());
        let mut wide = [0u8; 8];
        assert!(!Width::U64.write(-1, &mut wide));
        assert!(Width::I64.write(-1, &mut wide));
        assert_eq!(wide, (-1i64).to_ne_bytes());
    }
}
