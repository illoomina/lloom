//! Ordered JSON at the decision boundary. Uses serde's parser, not a JSON lexer.
//!
//! Python's `json.loads` retains the first position of a repeated key and replaces
//! its value with the last occurrence. We deliberately preserve that behavior at
//! every depth. Going through `serde_json::Value` first loses this information.

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl OrderedJson {
    /// Parse the original request bytes, before any unordered JSON conversion.
    pub fn from_slice(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Pretty state JSON used by d1-3B's default `json_only` renderer. Strings
    /// remain literal text; callers decide separately whether null omits state.
    pub fn state_text(&self) -> serde_json::Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            _ => serde_json::to_string_pretty(self),
        }
    }
}

impl Serialize for OrderedJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(v) => serializer.serialize_bool(*v),
            Self::Number(v) => v.serialize(serializer),
            Self::String(v) => serializer.serialize_str(v),
            Self::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(value)?;
                }
                seq.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = OrderedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(OrderedJson::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(OrderedJson::Number(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(OrderedJson::Number(value.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(OrderedJson::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(OrderedJson::String(value.into()))
            }
            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(OrderedJson::String(value))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(OrderedJson::Array(values))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values: Vec<(String, OrderedJson)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, OrderedJson>()? {
                    if let Some((_, previous)) = values.iter_mut().find(|(k, _)| *k == key) {
                        *previous = value;
                    } else {
                        values.push((key, value));
                    }
                }
                Ok(OrderedJson::Object(values))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::OrderedJson;

    #[test]
    fn raw_boundary_keeps_question_option_and_nested_state_order() {
        let input = br#"{"state":{"z":{"beta":"\u96ea\nline","alpha":1},"a":false},"questions":{"z":{"criteria":{"Z":null,"A":"a"}},"a":{}}}"#;
        let value = OrderedJson::from_slice(input).unwrap();
        assert_eq!(serde_json::to_string(&value).unwrap(),
            "{\"state\":{\"z\":{\"beta\":\"雪\\nline\",\"alpha\":1},\"a\":false},\"questions\":{\"z\":{\"criteria\":{\"Z\":null,\"A\":\"a\"}},\"a\":{}}}");
        assert_eq!(value.get("state").unwrap().state_text().unwrap(),
            "{\n  \"z\": {\n    \"beta\": \"雪\\nline\",\n    \"alpha\": 1\n  },\n  \"a\": false\n}");
    }

    #[test]
    fn duplicate_keys_replace_value_without_moving_first_position() {
        let value = OrderedJson::from_slice(br#"{"z":0,"a":{"b":1,"a":2,"b":3},"z":4}"#).unwrap();
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            "{\"z\":4,\"a\":{\"b\":3,\"a\":2}}"
        );
    }

    #[test]
    fn parser_rejects_trailing_data_and_preserves_literal_state_newlines() {
        assert!(OrderedJson::from_slice(b"{} []").is_err());
        let value = OrderedJson::from_slice(br#""one\n\u96ea""#).unwrap();
        assert_eq!(value.state_text().unwrap(), "one\n雪");
    }
}
