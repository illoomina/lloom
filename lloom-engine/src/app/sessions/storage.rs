//! Capacity-visible ownership of a message, without a JSON parser roundtrip.
//!
//! `Number` is retained unchanged. The pinned serde_json 1.0.151 feature graph
//! excludes `arbitrary_precision`: its Number is an inline integer/f64 enum,
//! not a heap string (number.rs:22-36). A feature/dependency change must re-audit
//! this assumption; the real allocator footprint and exact-value tests guard it.
//! Array/object buffers are boxed slices; strings/keys are boxed str, so every
//! retained allocation request is visible without estimating map internals.
use serde_json::{Map, Number, Value};
use std::mem::size_of;

#[derive(Clone)]
pub struct StoredMessage(StoredValue);

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

    /// Reconstruct a request-owned Value outside the session table mutex.
    pub fn value(&self) -> Value {
        self.0.value()
    }

    pub(super) fn heap_bytes(&self) -> usize {
        self.0.heap_bytes()
    }
}

/// Convert a detached, owned lookup snapshot to request/test JSON. Like
/// `StoredMessage::value`, call this only after releasing the table mutex.
/// Consuming moves string/key ownership while preserving Number/key/order
/// semantics. Container conversion can still allocate; no second retained
/// history or native/device owner is introduced.
impl From<StoredMessage> for Value {
    fn from(message: StoredMessage) -> Self {
        message.0.into_value()
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

    fn into_value(self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(value),
            Self::Number(value) => Value::Number(value),
            Self::String(value) => Value::String(value.into_string()),
            Self::Array(values) => Value::Array(
                values
                    .into_vec()
                    .into_iter()
                    .map(Self::into_value)
                    .collect(),
            ),
            Self::Object(values) => Value::Object(
                values
                    .into_vec()
                    .into_iter()
                    .map(|(key, value)| (key.into_string(), value.into_value()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_string_conversion_moves_payload_and_preserves_borrowed_api() {
        let text: Box<str> = "π\\\"\n-retained-payload".repeat(32).into_boxed_str();
        let pointer = text.as_ptr();
        let stored = StoredMessage(StoredValue::String(text));
        let borrowed = stored.value();
        let borrowed_again = stored.value();
        assert_eq!(
            borrowed, borrowed_again,
            "borrowed conversion is repeatable"
        );
        assert_ne!(borrowed.as_str().unwrap().as_ptr(), pointer);
        let owned = Value::from(stored);
        assert_eq!(owned, borrowed);
        assert_eq!(
            owned.as_str().unwrap().as_ptr(),
            pointer,
            "owned conversion copied the detached string payload"
        );
    }

    #[test]
    fn owned_nested_conversion_moves_keys_and_preserves_exact_values() {
        let key: Box<str> = "nested-π-key".into();
        let key_pointer = key.as_ptr();
        let payload: Box<str> = "nested-π-payload".repeat(32).into_boxed_str();
        let payload_pointer = payload.as_ptr();
        let floats = [-0.0, f64::from_bits(1), 51.248178375505404];
        let mut items = vec![StoredValue::String(payload)];
        items.extend(
            floats
                .into_iter()
                .map(|n| StoredValue::Number(Number::from_f64(n).expect("finite fixture"))),
        );
        items.extend([
            StoredValue::Number(Number::from(u64::MAX)),
            StoredValue::Number(Number::from(i64::MIN)),
            StoredValue::String("".into()),
            StoredValue::Array(Vec::new().into_boxed_slice()),
            StoredValue::Object(Vec::new().into_boxed_slice()),
            StoredValue::Null,
            StoredValue::Bool(true),
        ]);
        let stored = StoredMessage(StoredValue::Object(
            vec![(key, StoredValue::Array(items.into_boxed_slice()))].into_boxed_slice(),
        ));
        let borrowed = stored.value();
        let owned = Value::from(stored);
        assert_eq!(
            serde_json::to_string(&owned).unwrap(),
            serde_json::to_string(&borrowed).unwrap()
        );
        let (key, values) = owned.as_object().unwrap().iter().next().unwrap();
        assert_eq!(key, "nested-π-key");
        assert_eq!(
            key.as_ptr(),
            key_pointer,
            "owned conversion copied an object key"
        );
        let values = values.as_array().unwrap();
        assert_eq!(
            values[0].as_str().unwrap().as_ptr(),
            payload_pointer,
            "owned conversion copied a nested payload"
        );
        for (index, expected) in floats.into_iter().enumerate() {
            assert_eq!(
                values[index + 1].as_f64().unwrap().to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(values[4].as_u64(), Some(u64::MAX));
        assert_eq!(values[5].as_i64(), Some(i64::MIN));
        assert_eq!(values[6], "");
        assert!(values[7].as_array().unwrap().is_empty());
        assert!(values[8].as_object().unwrap().is_empty());
        assert!(values[9].is_null());
        assert_eq!(values[10], true);
    }
}
