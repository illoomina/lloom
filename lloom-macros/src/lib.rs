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

/// `#[derive(Config)]`.
///
/// The struct must have named fields. Fields tagged `#[axis(...)]` are the
/// engine's configuration axes: each one contributes to the `AXES` slice and
/// becomes a `clap` flag.
#[proc_macro_derive(Config, attributes(axis))]
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
                let doc = format!("Axis `{}`. Off unless set.", attr.name);
                quote! {
                    #[doc = #doc]
                    #[arg(long = #flag)]
                    pub #ident: bool,
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
                quote! {
                    #[doc = #doc]
                    #[arg(long = #flag)]
                    pub #ident: Option<String>,
                }
            }
        }
    });

    let apply_stmts = axes.iter().map(|(ident, _ty, attr)| match &attr.kind {
        AxisKind::Bool => quote! {
            base.#ident = args.#ident;
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
    let expanded = quote! {
        #(#axes_statics)*

        #[doc = #args_doc]
        #[derive(::clap::Args, Debug, Clone)]
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
