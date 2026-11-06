//! A lowered node's commands as functions of the step: each field that moves
//! with a named parameter becomes an exact affine patch
//! (`design/docs/planned-step-design.md` §5).
//!
//! A node is lowered once at a base point, and twice more per named
//! parameter with only that parameter moved. A field that moves is fitted as
//! `c0 + Σ cᵢ·pᵢ` from those samples and must reproduce all of them exactly.
//! Anything a patch cannot express refuses the node: a moved grid, binding
//! or structure, a moved field that is not a whole 4- or 8-byte scalar, a
//! read of a parameter the node did not name. Under `planned-validate` the
//! node is lowered once more with every named parameter at a held-out value,
//! and the fitted commands must predict that lowering field for field.

use std::cell::Cell;

use super::list::{Affine, CommandList, StepParam, StepValues, Width};
use super::lower::{LoweredArg, LoweredCommand, Lowering, NodeAlloc, Refusal, Refused};
use crate::kernels::metal::Grid;
use crate::metal_backend::MetalDevice;

/// How a named parameter is sampled: three distinct values for the fit, and
/// a fourth the fit has not seen for the held-out check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sweep {
    pub param: StepParam,
    pub at: [i64; 3],
    pub held_out: i64,
}

/// The step parameters one lowering of a node sees. Reading a parameter the
/// node did not name is recorded, and refuses the node: its dependence on
/// that parameter would never be sampled.
pub struct Probe<'a> {
    values: &'a StepValues,
    named: &'a [Sweep],
    undeclared: Cell<Option<StepParam>>,
}

impl<'a> Probe<'a> {
    /// The values `values`, of which the node named the parameters of `named`.
    pub fn new(values: &'a StepValues, named: &'a [Sweep]) -> Self {
        Self {
            values,
            named,
            undeclared: Cell::new(None),
        }
    }

    pub fn get(&self, param: StepParam) -> i64 {
        if !self.named.iter().any(|s| s.param == param) && self.undeclared.get().is_none() {
            self.undeclared.set(Some(param));
        }
        self.values.get(param)
    }
}

/// One declared node: the tag its commands carry, its outputs, and the
/// parameters its lowering may read.
pub struct Node<'a> {
    pub tag: u32,
    pub allocs: &'a [NodeAlloc],
    pub sweeps: &'a [Sweep],
}

/// Lower `node` at its sample points, fit what moved, and append its
/// commands to `list`. `base` gives every parameter's value at the base
/// point; the node's named parameters are taken from their sweeps instead.
/// Returns how many commands were appended.
pub fn lower_node(
    lowering: &Lowering,
    live: &MetalDevice,
    list: &mut CommandList,
    node: &Node<'_>,
    base: &StepValues,
    op: impl Fn(&MetalDevice, &Probe<'_>) -> crate::Result<()>,
) -> Result<usize, Refusal> {
    let refuse = |reason| Refusal {
        node: node.tag,
        reason,
    };
    for s in node.sweeps {
        let [a, b, c] = s.at;
        if a == b || b == c || a == c {
            return Err(refuse(Refused::Samples { param: s.param.0 }));
        }
        if s.param.0 as usize >= list.params() || s.param.0 as usize >= base.len() {
            return Err(refuse(Refused::List(format!(
                "parameter {} of {}",
                s.param.0,
                list.params().min(base.len())
            ))));
        }
    }
    let lower_at = |values: &StepValues| -> Result<Vec<LoweredCommand>, Refusal> {
        let probe = Probe::new(values, node.sweeps);
        let lowered = lowering.lower(live, node.tag, node.allocs, |d| op(d, &probe))?;
        match probe.undeclared.get() {
            Some(p) => Err(refuse(Refused::Undeclared { param: p.0 })),
            None => Ok(lowered),
        }
    };

    let mut origin = base.clone();
    for s in node.sweeps {
        origin.set(s.param, s.at[0]);
    }
    let first = lower_at(&origin)?;
    let mut samples = Vec::with_capacity(node.sweeps.len());
    for s in node.sweeps {
        let mut moved = [Vec::new(), Vec::new()];
        for (k, slot) in moved.iter_mut().enumerate() {
            let mut values = origin.clone();
            values.set(s.param, s.at[k + 1]);
            *slot = lower_at(&values)?;
        }
        samples.push(moved);
    }

    let start = list.len();
    let fitted = fit(&first, node.sweeps, &samples).map_err(refuse)?;
    let written = emit(list, node.tag, &first, &fitted).map_err(refuse);
    #[cfg(feature = "planned-validate")]
    let written = written.and_then(|()| {
        let mut held = origin.clone();
        for s in node.sweeps {
            held.set(s.param, s.held_out);
        }
        if node.sweeps.is_empty() {
            return Ok(());
        }
        let lowered = lower_at(&held)?;
        predicts(list, start, &held, &lowered).map_err(refuse)
    });
    if let Err(refusal) = written {
        list.truncate(start);
        return Err(refusal);
    }
    Ok(list.len() - start)
}

/// A field of the base lowering, and how it follows the step when it moves.
enum Field {
    Fixed,
    Offset(Affine),
    Bytes(Width, Affine),
}

/// Every field of every command, in argument order.
type Fitted = Vec<Vec<Field>>;

fn same_shape(a: &LoweredCommand, b: &LoweredCommand, command: usize) -> Result<(), Refused> {
    let structure = |what| Err(Refused::Structure { command, what });
    if !a.same_pipeline(b) {
        return structure("pipeline");
    }
    if a.grid != b.grid || a.threads != b.threads {
        return structure("grid");
    }
    if a.args.len() != b.args.len() {
        return structure("arguments");
    }
    for (x, y) in a.args.iter().zip(&b.args) {
        match (x, y) {
            (
                LoweredArg::Bytes { index: i, bytes: p },
                LoweredArg::Bytes { index: j, bytes: q },
            ) if i == j && p.len() == q.len() => {}
            (
                LoweredArg::Buffer {
                    index: i,
                    buffer: p,
                    write: w,
                    ..
                },
                LoweredArg::Buffer {
                    index: j,
                    buffer: q,
                    write: v,
                    ..
                },
            ) if i == j && p == q && w == v => {}
            (
                LoweredArg::ThreadgroupMemory { index: i, len: p },
                LoweredArg::ThreadgroupMemory { index: j, len: q },
            ) if i == j && p == q => {}
            (LoweredArg::Buffer { .. }, LoweredArg::Buffer { .. }) => return structure("binding"),
            (LoweredArg::ThreadgroupMemory { .. }, LoweredArg::ThreadgroupMemory { .. }) => {
                return structure("threadgroup memory")
            }
            _ => return structure("arguments"),
        }
    }
    Ok(())
}

/// The integer a moved field holds, read in `width`.
fn read(width: Width, bytes: &[u8]) -> i64 {
    match width {
        Width::U32 => i64::from(u32::from_ne_bytes(bytes.try_into().expect("4 bytes"))),
        Width::I32 => i64::from(i32::from_ne_bytes(bytes.try_into().expect("4 bytes"))),
        Width::U64 | Width::I64 => i64::from_ne_bytes(bytes.try_into().expect("8 bytes")),
    }
}

/// The exact affine function through a field's samples: `base` at the
/// origin, and per sweep its value at the sweep's second and third points.
/// `None` when no such function reproduces every sample.
fn line(base: i64, sweeps: &[Sweep], moved: &[[i64; 2]]) -> Option<Affine> {
    let mut terms = Vec::new();
    let mut c0 = i128::from(base);
    for (s, &[f1, f2]) in sweeps.iter().zip(moved) {
        let [a, b, c] = s.at.map(i128::from);
        let (d1, d2) = (
            i128::from(f1) - i128::from(base),
            i128::from(f2) - i128::from(base),
        );
        if d1 % (b - a) != 0 {
            return None;
        }
        let k = d1 / (b - a);
        if d2 != k * (c - a) {
            return None;
        }
        if k != 0 {
            terms.push((s.param, i64::try_from(k).ok()?));
            c0 -= k * a;
        }
    }
    Some(Affine::new(i64::try_from(c0).ok()?, &terms))
}

fn fit(
    first: &[LoweredCommand],
    sweeps: &[Sweep],
    samples: &[[Vec<LoweredCommand>; 2]],
) -> Result<Fitted, Refused> {
    for moved in samples.iter().flatten() {
        if moved.len() != first.len() {
            return Err(Refused::Structure {
                command: first.len().min(moved.len()),
                what: "dispatch count",
            });
        }
        for (command, (a, b)) in first.iter().zip(moved).enumerate() {
            same_shape(a, b, command)?;
        }
    }
    let mut fitted = Vec::with_capacity(first.len());
    for (command, base) in first.iter().enumerate() {
        let mut fields = Vec::with_capacity(base.args.len());
        for (arg, at) in base.args.iter().enumerate() {
            let sampled = |f: &dyn Fn(&LoweredArg) -> i64| -> Vec<[i64; 2]> {
                samples
                    .iter()
                    .map(|[p, q]| [f(&p[command].args[arg]), f(&q[command].args[arg])])
                    .collect()
            };
            let field = match at {
                LoweredArg::Buffer { index, offset, .. } => {
                    let offset_of = |a: &LoweredArg| match a {
                        LoweredArg::Buffer { offset, .. } => *offset as i64,
                        _ => unreachable!("shapes checked"),
                    };
                    let moved = sampled(&offset_of);
                    let o0 = *offset as i64;
                    if moved.iter().flatten().all(|&o| o == o0) {
                        Field::Fixed
                    } else {
                        let param = moving(sweeps, &moved, o0);
                        Field::Offset(line(o0, sweeps, &moved).ok_or(Refused::NonAffine {
                            command,
                            index: *index,
                            param,
                        })?)
                    }
                }
                LoweredArg::Bytes { index, bytes } => {
                    let all_same = samples
                        .iter()
                        .flatten()
                        .all(|l| match &l[command].args[arg] {
                            LoweredArg::Bytes { bytes: b, .. } => b == bytes,
                            _ => unreachable!("shapes checked"),
                        });
                    if all_same {
                        Field::Fixed
                    } else {
                        let widths: &[Width] = match bytes.len() {
                            4 => &[Width::U32, Width::I32],
                            8 => &[Width::I64],
                            len => {
                                return Err(Refused::Width {
                                    command,
                                    index: *index,
                                    len,
                                })
                            }
                        };
                        let mut found = None;
                        let mut param = 0;
                        for &width in widths {
                            let value_of = |a: &LoweredArg| match a {
                                LoweredArg::Bytes { bytes, .. } => read(width, bytes),
                                _ => unreachable!("shapes checked"),
                            };
                            let moved = sampled(&value_of);
                            let v0 = read(width, bytes);
                            param = moving(sweeps, &moved, v0);
                            if let Some(f) = line(v0, sweeps, &moved) {
                                found = Some(Field::Bytes(width, f));
                                break;
                            }
                        }
                        found.ok_or(Refused::NonAffine {
                            command,
                            index: *index,
                            param,
                        })?
                    }
                }
                LoweredArg::ThreadgroupMemory { .. } => Field::Fixed,
            };
            fields.push(field);
        }
        fitted.push(fields);
    }
    Ok(fitted)
}

/// The first parameter whose samples moved a field: the one a refusal names.
fn moving(sweeps: &[Sweep], moved: &[[i64; 2]], base: i64) -> u16 {
    sweeps
        .iter()
        .zip(moved)
        .find(|(_, m)| m.iter().any(|&v| v != base))
        .map_or(0, |(s, _)| s.param.0)
}

fn emit(
    list: &mut CommandList,
    tag: u32,
    first: &[LoweredCommand],
    fitted: &Fitted,
) -> Result<(), Refused> {
    let list_err = |e: crate::kernels::MetalKernelError| Refused::List(e.to_string());
    for (command, fields) in first.iter().zip(fitted) {
        let mut c = list.command(command.pipeline.clone(), tag);
        for (arg, field) in command.args.iter().zip(fields) {
            match (arg, field) {
                (LoweredArg::Bytes { index, bytes }, Field::Fixed) => {
                    c.bytes(*index as usize, bytes);
                }
                (LoweredArg::Bytes { index, bytes }, Field::Bytes(width, value)) => {
                    let id = c.bytes(*index as usize, bytes);
                    c.patch_bytes(id, 0, *width, value.clone())
                        .map_err(list_err)?;
                }
                (
                    LoweredArg::Buffer {
                        index,
                        buffer,
                        offset,
                        write,
                    },
                    field,
                ) => {
                    let id = c.buffer(*index as usize, *buffer, *offset, *write);
                    if let Field::Offset(value) = field {
                        c.patch_offset(id, value.clone()).map_err(list_err)?;
                    }
                }
                (LoweredArg::ThreadgroupMemory { index, len }, _) => {
                    c.threadgroup_memory(*index as usize, *len as usize);
                }
                (LoweredArg::Bytes { .. }, Field::Offset(_)) => {
                    unreachable!("an offset field is fitted on a buffer argument")
                }
            }
        }
        match command.grid {
            Grid::Threadgroups(groups) => c.dispatch_threadgroups(groups, command.threads),
            Grid::Threads(grid) => c.dispatch_threads(grid, command.threads),
        }
    }
    Ok(())
}

/// Whether the commands from `start` in `list`, their patches evaluated at
/// `values`, are exactly `lowered`: the held-out check, and the reference a
/// fit's tests hold it against.
#[cfg(any(test, feature = "planned-validate"))]
pub fn predicts(
    list: &CommandList,
    start: usize,
    values: &StepValues,
    lowered: &[LoweredCommand],
) -> Result<(), Refused> {
    use super::list::{apply_bytes, offset_at, Arg, PatchTarget};
    use super::lower::same_pipeline;
    let commands = list.commands().get(start..).unwrap_or_default();
    if commands.len() != lowered.len() {
        return Err(Refused::Structure {
            command: commands.len().min(lowered.len()),
            what: "dispatch count",
        });
    }
    for (command, (planned, seen)) in commands.iter().zip(lowered).enumerate() {
        let held = |index| Refused::HeldOut { command, index };
        if !same_pipeline(planned.pipeline(), &seen.pipeline)
            || planned.grid() != seen.grid
            || planned.threads() != seen.threads
        {
            return Err(Refused::Structure {
                command,
                what: "pipeline or grid",
            });
        }
        let args = list.args(planned);
        if args.len() != seen.args.len() {
            return Err(Refused::Structure {
                command,
                what: "arguments",
            });
        }
        let patches = list.patches(planned);
        for (i, (arg, seen)) in args.iter().zip(&seen.args).enumerate() {
            let id = planned.args.start + i as u32;
            let mine = patches.iter().filter(|p| p.arg == id);
            match (*arg, seen) {
                (
                    Arg::Bytes { index, start, len },
                    LoweredArg::Bytes {
                        index: j,
                        bytes: theirs,
                    },
                ) => {
                    let mut bytes = list.bytes(start, len).to_vec();
                    for p in mine {
                        apply_bytes(p, values, &mut bytes).map_err(|_| held(index))?;
                    }
                    if index != *j || &bytes != theirs {
                        return Err(held(index));
                    }
                }
                (
                    Arg::Buffer {
                        index,
                        buffer,
                        offset,
                        write,
                    },
                    LoweredArg::Buffer {
                        index: j,
                        buffer: b,
                        offset: o,
                        write: w,
                    },
                ) => {
                    let mut at = offset;
                    for p in mine {
                        debug_assert_eq!(p.target(), PatchTarget::Offset);
                        at = offset_at(p, values).map_err(|_| held(index))?;
                    }
                    if index != *j || buffer != *b || at != *o || write != *w {
                        return Err(held(index));
                    }
                }
                (
                    Arg::ThreadgroupMemory { index, len },
                    LoweredArg::ThreadgroupMemory { index: j, len: l },
                ) if index == *j && len == *l => {}
                (Arg::Bytes { index, .. }, _)
                | (Arg::Buffer { index, .. }, _)
                | (Arg::ThreadgroupMemory { index, .. }, _) => return Err(held(index)),
            }
        }
    }
    Ok(())
}
