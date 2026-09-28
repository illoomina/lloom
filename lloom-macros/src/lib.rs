//! Derive macros for the lloom engine.
//!
//! `#[derive(Config)]` emits, per annotated struct, two products that
//! `RESTRUCTURING_DESIGN.md` §1.6 locks:
//!
//! 1. one `#[linkme::distributed_slice(AXES)]` static per `#[axis]`-tagged
//!    field, contributing an `AxisMeta` into the crate-wide `AXES` slice — the
//!    within-crate registry the mirror test reconciles against `REGISTRY`;
//! 2. a companion `<Struct>Args` that derives `clap::Args`, so `--attn` /
//!    `--kv-quant` and every other axis flag falls out of the same `#[axis]`
//!    tags rather than being hand-rolled per harness. It is a companion rather
//!    than `impl clap::Args for <Struct>` because that trait needs
//!    `FromArgMatches` and the config cannot be built from CLI matches alone —
//!    its shape fields come from the checkpoint.
//!
//! The `#[axis]` attribute carries the registry name, the variant set and the
//! default *explicitly* rather than introspecting the field type. The axis
//! enums are a mix — some are unit enums (`AttnImpl`), some carry data and are
//! parsed by a hand-written `parse(&str)` (`ConvState::SlidingRing{k,slack}`) —
//! so a type-introspecting macro would have to special-case each shape. An
//! explicit attribute is the shape `.bench/configurations.md` §1 already uses
//! (name, variants, default as prose) and keeps the macro one code path.
//!
//! A struct whose fields carry `#[tiny(...)]` also gets the tiny-model
//! surface (`docs/porting-models.md` §21.8): the `TINY` field table, the axis
//! accessors a shape search mutates through, `tiny_derive`, which
//! recomputes the derived fields from the axes, and the same four as
//! `lloom::testing::tiny::TinyConfig`, which a campaign is generic over.
//! Emitted only when some field is tagged, so a config without `#[tiny]`
//! expands exactly as before.

mod options;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Fields, LitStr, Token, Type,
};

/// The parsed `#[axis(...)]` attribute on one field.
struct AxisAttr {
    /// The registry name, e.g. `"AttnImpl"` — must match a `REGISTRY` entry.
    name: String,
    /// The CLI flag long name, e.g. `"attn"` for `--attn`.
    flag: String,
    /// The variant set, rendered into `AxisMeta` and into the flag's help.
    variants: Vec<String>,
    /// The default arm, e.g. `"Generic"`.
    default: String,
    /// How the flag string maps to the field: `Parsed` routes a string value
    /// through a named `parse` fn; `Bool` is a plain switch.
    kind: AxisKind,
}

#[derive(PartialEq)]
enum AxisKind {
    /// A string-valued flag parsed into the field type via a named function.
    Parsed(String),
    /// A `bool` switch.
    Bool,
}

impl Parse for AxisAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut flag = None;
        let mut variants = Vec::new();
        let mut default = None;
        let mut parser = None;
        let mut is_bool = false;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "bool" => {
                    is_bool = true;
                }
                "name" => {
                    input.parse::<Token![=]>()?;
                    name = Some(input.parse::<LitStr>()?.value());
                }
                "flag" => {
                    input.parse::<Token![=]>()?;
                    flag = Some(input.parse::<LitStr>()?.value());
                }
                "default" => {
                    input.parse::<Token![=]>()?;
                    default = Some(input.parse::<LitStr>()?.value());
                }
                "parser" => {
                    input.parse::<Token![=]>()?;
                    parser = Some(input.parse::<LitStr>()?.value());
                }
                "variants" => {
                    input.parse::<Token![=]>()?;
                    let content;
                    syn::bracketed!(content in input);
                    let items = content.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
                    variants = items.into_iter().map(|s| s.value()).collect();
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[axis] key `{other}`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let name =
            name.ok_or_else(|| syn::Error::new(input.span(), "#[axis] requires `name = \"...\"`"))?;
        let flag = flag.unwrap_or_else(|| name.to_lowercase());
        let default = default.unwrap_or_default();
        let kind = if is_bool {
            AxisKind::Bool
        } else {
            AxisKind::Parsed(parser.unwrap_or_else(|| "parse".to_string()))
        };

        Ok(AxisAttr {
            name,
            flag,
            variants,
            default,
            kind,
        })
    }
}

/// The parsed `#[tiny(...)]` attribute on one field.
#[derive(Default)]
struct TinyAttr {
    /// A searched dimension (`usize`).
    axis: bool,
    /// Kept at the template's value.
    fixed: bool,
    /// Every layer kind the template has must survive a mutation.
    cover: bool,
    /// Recomputed from the other fields by this expression over their names.
    derived: Option<syn::Expr>,
    /// The largest value a search may set.
    max: Option<usize>,
}

impl Parse for TinyAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut t = TinyAttr::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "axis" => t.axis = true,
                "fixed" => t.fixed = true,
                "cover" => t.cover = true,
                "derived" => {
                    input.parse::<Token![=]>()?;
                    let s = input.parse::<LitStr>()?;
                    t.derived = Some(s.parse::<syn::Expr>()?);
                }
                "max" => {
                    input.parse::<Token![=]>()?;
                    t.max = Some(input.parse::<syn::LitInt>()?.base10_parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[tiny] key `{other}` (axis, fixed, cover, derived, max)"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        let roles = [t.axis, t.fixed, t.derived.is_some()]
            .iter()
            .filter(|b| **b)
            .count();
        if roles != 1 {
            return Err(
                input.error("#[tiny] takes exactly one of `axis`, `fixed`, `derived = \"...\"`")
            );
        }
        if t.max.is_some() && !t.axis {
            return Err(input.error("#[tiny] `max` applies to an `axis` only"));
        }
        Ok(t)
    }
}

/// The tiny-model surface for the `#[tiny]`-tagged fields; empty when none is.
fn tiny_surface(
    struct_ident: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, Token![,]>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut tagged: Vec<(syn::Ident, Type, TinyAttr)> = Vec::new();
    for field in fields {
        let Some(ident) = field.ident.clone() else {
            continue;
        };
        for attr in &field.attrs {
            if attr.path().is_ident("tiny") {
                tagged.push((ident.clone(), field.ty.clone(), attr.parse_args()?));
            }
        }
    }
    if tagged.is_empty() {
        return Ok(quote! {});
    }
    for (ident, ty, t) in &tagged {
        if t.axis && quote!(#ty).to_string() != "usize" {
            return Err(syn::Error::new_spanned(
                ty,
                format!("#[tiny(axis)] `{ident}` must be `usize`"),
            ));
        }
    }
    let rows = tagged.iter().map(|(ident, _, t)| {
        let name = ident.to_string();
        let role = if t.axis {
            quote!(crate::testing::tiny::TinyRole::Axis)
        } else if t.fixed {
            quote!(crate::testing::tiny::TinyRole::Fixed)
        } else {
            let e = t.derived.as_ref().expect("one role");
            let text = quote!(#e).to_string();
            quote!(crate::testing::tiny::TinyRole::Derived(#text))
        };
        let cover = t.cover;
        let max = match t.max {
            Some(m) => quote!(::std::option::Option::Some(#m)),
            None => quote!(::std::option::Option::None),
        };
        quote! {
            crate::testing::tiny::TinyField { name: #name, role: #role, cover: #cover, max: #max }
        }
    });
    let axes: Vec<&syn::Ident> = tagged
        .iter()
        .filter(|(_, _, t)| t.axis)
        .map(|(i, _, _)| i)
        .collect();
    let axis_names: Vec<String> = axes.iter().map(|i| i.to_string()).collect();
    // Every `#[tiny]` field bound by name, so a derived expression reads its
    // inputs as plain identifiers; each derived field is rebound after it is
    // computed. Untagged fields are not bound: a derived dimension depends on
    // the geometry only, and an untagged field may be `#[cfg]`-gated.
    let all: Vec<&syn::Ident> = tagged.iter().map(|(i, _, _)| i).collect();
    let derived = tagged
        .iter()
        .filter_map(|(i, _, t)| t.derived.as_ref().map(|e| (i, e)));
    let derive_stmts = derived.map(|(i, e)| {
        quote! {
            self.#i = #e;
            let #i = ::std::clone::Clone::clone(&self.#i);
        }
    });
    Ok(quote! {
        impl #struct_ident {
            /// The `#[tiny]` annotations, in field order (`docs/porting-models.md` §21.8).
            pub const TINY: &'static [crate::testing::tiny::TinyField] = &[#(#rows),*];

            /// The value of the `#[tiny(axis)]` field `name`.
            pub fn tiny_axis(&self, name: &str) -> ::std::option::Option<usize> {
                match name {
                    #(#axis_names => ::std::option::Option::Some(self.#axes),)*
                    _ => ::std::option::Option::None,
                }
            }

            /// Set the `#[tiny(axis)]` field `name`; `false` if there is none.
            /// The derived fields are stale until [`Self::tiny_derive`].
            pub fn set_tiny_axis(&mut self, name: &str, value: usize) -> bool {
                match name {
                    #(#axis_names => { self.#axes = value; true })*
                    _ => false,
                }
            }

            /// Recompute every `#[tiny(derived = "...")]` field, in field order.
            #[allow(unused_variables, clippy::redundant_clone)]
            pub fn tiny_derive(&mut self) {
                #(let #all = ::std::clone::Clone::clone(&self.#all);)*
                #(#derive_stmts)*
            }
        }

        impl crate::testing::tiny::TinyConfig for #struct_ident {
            fn fields() -> &'static [crate::testing::tiny::TinyField] {
                Self::TINY
            }
            fn axis(&self, name: &str) -> ::std::option::Option<usize> {
                self.tiny_axis(name)
            }
            fn set_axis(&mut self, name: &str, value: usize) -> bool {
                self.set_tiny_axis(name, value)
            }
            fn derive(&mut self) {
                self.tiny_derive()
            }
        }
    })
}

/// `#[derive(Config)]`.
///
/// The struct must have named fields. Fields tagged `#[axis(...)]` are the
/// engine's configuration axes: each one contributes to the `AXES` slice and
/// becomes a `clap` flag. Fields tagged `#[tiny(...)]` describe the family's
/// tiny-model geometry (see the crate docs).
#[proc_macro_derive(Config, attributes(axis, tiny))]
pub fn derive_config(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_ident = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    struct_ident,
                    "#[derive(Config)] requires named fields",
                )
                .to_compile_error()
                .into()
            }
        },
        _ => {
            return syn::Error::new_spanned(struct_ident, "#[derive(Config)] requires a struct")
                .to_compile_error()
                .into()
        }
    };

    // Collect the axis fields.
    let mut axes: Vec<(syn::Ident, Type, AxisAttr)> = Vec::new();
    for field in fields {
        let Some(ident) = field.ident.clone() else {
            continue;
        };
        for attr in &field.attrs {
            if !attr.path().is_ident("axis") {
                continue;
            }
            let parsed: AxisAttr = match attr.parse_args() {
                Ok(a) => a,
                Err(e) => return e.to_compile_error().into(),
            };
            axes.push((ident.clone(), field.ty.clone(), parsed));
        }
    }

    // 1. AXES contributions: one distributed_slice static per axis field.
    let axes_statics = axes.iter().enumerate().map(|(i, (ident, _ty, attr))| {
        let static_ident = format_ident!(
            "__LLOOM_AXIS_{}_{}",
            struct_ident.to_string().to_uppercase(),
            i
        );
        let name = &attr.name;
        let default = &attr.default;
        let variants = &attr.variants;
        let _ = ident;
        // linkme has no wasm32 backend; on wasm the AXES slice is empty and gets
        // no contributions (the registry is host-only, #656).
        quote! {
            #[cfg(not(target_arch = "wasm32"))]
            #[::linkme::distributed_slice(crate::axes::AXES)]
            static #static_ident: crate::axes::AxisMeta = crate::axes::AxisMeta {
                name: #name,
                variants: &[#(#variants),*],
                default: #default,
            };
        }
    });

    // 2. The clap flags — a generated companion `#struct_identArgs` that derives
    //    clap's own `Args`, embedded by a harness with `#[command(flatten)]`.
    //    Not `impl clap::Args for #struct_ident`: that trait needs
    //    `FromArgMatches`, and the config cannot be built from CLI matches alone
    //    (its shape fields come from the checkpoint). `apply` mutates a base
    //    config from the parsed flags, which is how the harnesses already work
    //    (parse a flag, mutate the field).
    let args_ident = format_ident!("{}Args", struct_ident);

    let arg_fields = axes.iter().map(|(ident, ty, attr)| {
        let flag = &attr.flag;
        let variants_help = attr.variants.join(", ");
        match &attr.kind {
            AxisKind::Bool => {
                let doc = format!("Axis `{}`: `--{flag}` / `--{flag}=false`; absent keeps the preset's arm.", attr.name);
                let id = format!("{}.{}", struct_ident, ident);
                quote! {
                    #[doc = #doc]
                    #[arg(id = #id, long = #flag, num_args = 0..=1, default_missing_value = "true", action = ::clap::ArgAction::Set, value_parser = ::clap::value_parser!(bool), value_name = "BOOL")]
                    pub #ident: Option<bool>,
                }
            }
            AxisKind::Parsed(_) => {
                // The axis arms are `{variants_help}`; the exact flag spellings
                // the parser accepts are that type's `parse`. Help names the
                // axis and its arms rather than claiming a spelling set, so it
                // cannot drift from the parser.
                let doc = format!("Axis `{}` (arms: {}).", attr.name, variants_help);
                // Optional so "flag absent" means "keep the base config's value"
                // rather than forcing a default the base already carries.
                let _ = ty;
                let id = format!("{}.{}", struct_ident, ident);
                quote! {
                    #[doc = #doc]
                    #[arg(id = #id, long = #flag)]
                    pub #ident: Option<String>,
                }
            }
        }
    });

    let apply_stmts = axes.iter().map(|(ident, _ty, attr)| match &attr.kind {
        AxisKind::Bool => quote! {
            if let Some(v) = args.#ident {
                base.#ident = v;
            }
        },
        AxisKind::Parsed(parser) => {
            let parse_fn: proc_macro2::TokenStream =
                parser.parse().unwrap_or_else(|_| quote!(parse));
            let flag = &attr.flag;
            quote! {
                if let Some(raw) = &args.#ident {
                    base.#ident = #parse_fn(raw)
                        .map_err(|e| format!("--{}: {}", #flag, e))?;
                }
            }
        }
    });

    // 3. The other front-ends, from the same tags: `set_axis(flag, value)`
    //    (the dotted key and `--set`), `merge_axes_json` (a `{flag: value}`
    //    object), `axes_line()` (the resolved arms for a report line, read
    //    back from the fields — §7.1a-i), `AXES_META` (the knobs table), and
    //    `ConfigArgs::overrides()` (the flags that were given, as pairs).
    let set_arms = axes.iter().map(|(ident, _ty, attr)| {
        let flag = &attr.flag;
        match &attr.kind {
            AxisKind::Bool => quote! {
                #flag => {
                    self.#ident = ::lloom_tensor::options::parse_bool(value)
                        .map_err(|e| format!("--{}: {}", #flag, e))?;
                    Ok(())
                }
            },
            AxisKind::Parsed(parser) => {
                let parse_fn: proc_macro2::TokenStream =
                    parser.parse().unwrap_or_else(|_| quote!(parse));
                quote! {
                    #flag => {
                        self.#ident = #parse_fn(value).map_err(|e| format!("--{}: {}", #flag, e))?;
                        Ok(())
                    }
                }
            }
        }
    });
    let flag_list = axes.iter().map(|(_, _, a)| a.flag.clone()).collect::<Vec<_>>().join(", ");
    let line_parts = axes.iter().map(|(ident, _ty, attr)| {
        let name = &attr.name;
        quote! { parts.push(format!("{}={:?}", #name, self.#ident)); }
    });
    let meta_items = axes.iter().map(|(ident, _ty, attr)| {
        let flag = &attr.flag;
        let name = &attr.name;
        let variants = &attr.variants;
        let default = &attr.default;
        let doc = format!("Axis `{name}` (default {default}).");
        let is_bool = attr.kind == AxisKind::Bool;
        let ty = format!("{}", quote!(#ident)).replace(' ', "");
        let _ = ty;
        quote! {
            ::lloom_tensor::options::KnobMeta {
                key: #flag, flag: #flag, env: ::std::option::Option::None, doc: #doc,
                variants: &[#(#variants),*], ty: #name, nested: ::std::option::Option::None,
                is_bool: #is_bool,
            }
        }
    });
    let override_stmts = axes.iter().map(|(ident, _ty, attr)| {
        let flag = &attr.flag;
        match &attr.kind {
            AxisKind::Bool => quote! {
                if let Some(v) = self.#ident { out.push((#flag, v.to_string())); }
            },
            AxisKind::Parsed(_) => quote! {
                if let Some(v) = &self.#ident { out.push((#flag, v.clone())); }
            },
        }
    });

    let args_doc = format!(
        "clap flags for the axis fields of `{struct_ident}`.\n\n\
         A harness embeds this with `#[command(flatten)]` and calls [`{args_ident}::apply`] \
         onto a base config built from the checkpoint. The flags fall out of the `#[axis]` \
         tags on `{struct_ident}` — `--attn`, `--kv-quant`, etc. — rather than being \
         hand-rolled per harness (RESTRUCTURING_DESIGN.md §1.6).\n\n\
         `Config` itself is deliberately NOT `impl clap::Args`: `clap::Args` requires \
         `FromArgMatches`, and `Config` cannot be built from CLI matches alone — its shape \
         fields (`hidden_size`, `layer_types`, …) come from the checkpoint. The companion \
         struct carries only the axes, which is what a run actually selects."
    );
    let tiny = match tiny_surface(struct_ident, fields) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };
    let expanded = quote! {
        #(#axes_statics)*

        #tiny

        #[doc = #args_doc]
        #[derive(::clap::Args, Debug, Clone, Default)]
        pub struct #args_ident {
            #(#arg_fields)*
        }

        impl #args_ident {
            /// Apply the parsed flags onto a base config, leaving unset axes at
            /// the base's value. Returns the flag name and the parse error on a
            /// bad value.
            pub fn apply(&self, base: &mut #struct_ident) -> ::std::result::Result<(), ::std::string::String> {
                let args = self;
                #(#apply_stmts)*
                Ok(())
            }

            /// The flags that were given, as `(flag, value)` pairs in the
            /// flag spelling — what a dotted `--set` or a JSON group carries.
            pub fn overrides(&self) -> ::std::vec::Vec<(&'static str, ::std::string::String)> {
                let mut out = ::std::vec::Vec::new();
                #(#override_stmts)*
                out
            }
        }

        impl #struct_ident {
            /// The axes as knobs (flag, arms, default), for the knobs table.
            pub const AXES_META: &'static [::lloom_tensor::options::KnobMeta] = &[#(#meta_items),*];

            /// Set one axis by its flag name, the value in its flag spelling
            /// (`set_axis("kv-append", "paged")`).
            pub fn set_axis(&mut self, flag: &str, value: &str) -> ::std::result::Result<(), ::std::string::String> {
                match flag {
                    #(#set_arms)*
                    other => Err(format!("unknown axis `{}` (axes: {})", other, #flag_list)),
                }
            }

            /// Merge a `{flag: value}` object onto the axes; values are strings
            /// in their flag spelling or bools.
            pub fn merge_axes_json(&mut self, json: &::serde_json::Value) -> ::std::result::Result<(), ::std::string::String> {
                let Some(map) = json.as_object() else {
                    return Err("model axes: expected a JSON object of {flag: value}".to_string());
                };
                for (k, v) in map {
                    let value = match v {
                        ::serde_json::Value::String(s) => s.clone(),
                        ::serde_json::Value::Bool(b) => b.to_string(),
                        ::serde_json::Value::Number(n) => n.to_string(),
                        other => return Err(format!("axis `{}`: expected a string, got {}", k, other)),
                    };
                    self.set_axis(k, &value)?;
                }
                Ok(())
            }

            /// The resolved arms, `Name=Arm …`, read back from the fields.
            pub fn axes_line(&self) -> ::std::string::String {
                let mut parts: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                #(#line_parts)*
                parts.join(" ")
            }
        }
    };

    expanded.into()
}

/// `#[derive(Options)]` — an option group with every front-end generated from
/// one field list: the clap companion `<Struct>Args` + `apply`, the dotted
/// `set`, `merge_json`, `from_env` and the `META` table. See `options.rs`.
#[proc_macro_derive(Options, attributes(opt, options))]
pub fn derive_options(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    options::derive(input).into()
}
