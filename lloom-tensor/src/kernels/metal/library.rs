use crate::kernels::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLDataType, MTLFunction, MTLFunctionConstantValues, MTLLibrary};
use std::{ffi::c_void, ptr};

#[derive(Clone, Debug)]
pub struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLLibrary>>) -> Library {
        Library { raw }
    }

    /// Every function the COMPILED library carries, as Metal reports it.
    ///
    /// This is the instantiated set read from the artifact rather than from the
    /// source that produced it — `DESIGN.md` §8.1b's argument for a checked
    /// registry over a generator, applied to enumeration instead of lookup.
    ///
    /// It matters because a source-side count cannot be taken. Eighteen of the
    /// twenty `.metal` files instantiate through a macro whose `[[host_name]]`
    /// argument is a *variable* (`reduce.metal:790`, `gemv.metal:1102`), so a
    /// grep for a literal name is structurally blind to every family a decode
    /// token dispatches and sees only `quantized.metal`, which spells its names
    /// inline (lloom #517, correcting #469's 133).
    ///
    /// Reading it compiles a library and enqueues no GPU work, so it needs no
    /// arbiter lease — §13.1i records that compiling a library and creating a
    /// pipeline use the driver rather than the queue `lloom-arb` arbitrates.
    pub fn function_names(&self) -> Vec<String> {
        self.raw
            .functionNames()
            .iter()
            .map(|name| name.to_string())
            .collect()
    }

    pub fn get_function(
        &self,
        name: &str,
        constant_values: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let function = match constant_values {
            Some(constant_values) => self
                .raw
                .newFunctionWithName_constantValues_error(
                    &NSString::from_str(name),
                    &constant_values.function_constant_values().raw,
                )
                .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?,
            None => self
                .raw
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(MetalKernelError::LoadFunctionError(name.to_string()))?,
        };

        Ok(Function { raw: function })
    }
}

pub struct Function {
    raw: Retained<ProtocolObject<dyn MTLFunction>>,
}

impl AsRef<ProtocolObject<dyn MTLFunction>> for Function {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLFunction> {
        &self.raw
    }
}

pub struct FunctionConstantValues {
    raw: Retained<MTLFunctionConstantValues>,
}

impl FunctionConstantValues {
    pub fn new() -> FunctionConstantValues {
        FunctionConstantValues {
            raw: MTLFunctionConstantValues::new(),
        }
    }

    pub fn set_constant_value_at_index<T>(&self, value: &T, dtype: MTLDataType, index: usize) {
        let value = ptr::NonNull::new(value as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setConstantValue_type_atIndex(value, dtype, index) }
    }
}

impl Default for FunctionConstantValues {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            // usize is usually u64 aka ulong, but can be u32 on 32-bit systems.
            // https://developer.apple.com/documentation/objectivec/nsuinteger
            Value::USize(_) => MTLDataType::ULong,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
            }
        }
        f
    }
}
