//! Capacity-visible ownership of a message, without a JSON parser roundtrip.
use serde_json::{Map, Number, Value};
use std::mem::size_of;

#[derive(Clone)]
pub(super) struct StoredMessage(StoredValue);

#[derive(Clone)]
enum StoredValue {
    Null,
    Bool(bool),
    Number(Number),
    String(Box<str>),
    Array(Box<[StoredValue]>),
    Object(Box<[(Box<str>, StoredValue)]>),
}

impl StoredMessage {
    pub(super) fn new(value: Value) -> Self {
        Self(StoredValue::new(value))
    }

    pub(super) fn value(&self) -> Value {
        self.0.value()
    }

    pub(super) fn heap_bytes(&self) -> usize {
        self.0.heap_bytes()
    }
}

impl StoredValue {
    fn new(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value.into_boxed_str()),
            Value::Array(values) => Self::Array(values.into_iter().map(Self::new).collect()),
            Value::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key.into_boxed_str(), Self::new(value)))
                    .collect(),
            ),
        }
    }

    fn value(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.to_string()),
            Self::Array(values) => Value::Array(values.iter().map(Self::value).collect()),
            Self::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.value()))
                    .collect::<Map<_, _>>(),
            ),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Null | Self::Bool(_) | Self::Number(_) => 0,
            Self::String(value) => value.len(),
            Self::Array(values) => {
                values.len() * size_of::<Self>()
                    + values.iter().map(Self::heap_bytes).sum::<usize>()
            }
            Self::Object(values) => {
                values.len() * size_of::<(Box<str>, Self)>()
                    + values
                        .iter()
                        .map(|(key, value)| key.len() + value.heap_bytes())
                        .sum::<usize>()
            }
        }
    }
}
