//! Tiny models (`docs/porting-models.md` §21.8): the per-field annotations a
//! family's config carries (`#[tiny(...)]` on `#[derive(Config)]`), which the
//! shape campaign in `lloom-fuzz` reads to know what it may mutate.

/// What the shape search may do with one config field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TinyRole {
    /// A searched dimension.
    Axis,
    /// Kept at the template's value.
    Fixed,
    /// Recomputed from the other fields; the text is the expression.
    Derived(&'static str),
}

/// One `#[tiny]`-tagged field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyField {
    pub name: &'static str,
    pub role: TinyRole,
    /// Every layer kind the template has must survive a mutation.
    pub cover: bool,
    /// The largest value a search may set (an axis only).
    pub max: Option<usize>,
}
