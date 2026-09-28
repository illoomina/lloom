//! `#[derive(Options)]` — one option group, every front-end.
//!
//! An option group is a plain struct (`Default` + serde) whose fields are the
//! knobs a run selects. The derive emits, from the field list and the
//! `#[opt(...)]` tags, every way a value can reach it — so the flags, the
//! dotted keys, the JSON, the env aliases and the printed table are ONE
//! source of truth and cannot drift from each other:
//!
//! 1. `<Struct>Args`: a `clap::Args` companion (`--<flag>` per field, nested
//!    groups flattened), and `apply(&self, base: &mut Struct)` that writes
//!    only the flags that were given — "absent" keeps the base's value, so
//!    the layers compose (defaults < checkpoint < env < file < json < set <
//!    flags);
//! 2. `Struct::set(&mut self, key, value)`: a dotted-key setter
//!    (`decode.readback.eos = gpubatch`) for `--set` and config files;
//! 3. `Struct::merge_json(&mut self, &serde_json::Value)`: the vLLM /
//!    transformers shape (`--decode-config '{...}'`), field by field so a
//!    partial object overrides only what it names;
//! 4. `Struct::from_env(&mut self)`: the env aliases (`#[opt(env = "…")]`),
//!    kept so every script that sets `LLOOM_*` keeps working;
//! 5. `Struct::META`: the static table (`KnobMeta`) the `lloom knobs` command
//!    prints and the drift test reads — key, flag, env, doc, arms.
//!
//! Field tags:
//!
//! ```text
//! #[opt]                                     a value field: FromStr + serde
//! #[opt(flag = "kv-append")]                 the long flag (default: the field name, kebab-case)
//! #[opt(short = 't', alias = "temp")]         a short flag, an alias for the long one
//! #[opt(bool)]                               a switch: `--x` / `--x=false`
//! #[opt(env = "LLOOM_X")]                    an env alias read by `from_env`
//! #[opt(variants = ["a", "b"])]              the arms, for help and the table
//! #[opt(parser = "parse_spark")]             `fn(&str) -> Result<T, String>` instead of FromStr
//! #[opt(nested)]                             a sub-group: its own derive, flattened
//! #[opt(nested, flag = "readback", parser = "f")]  …plus a compound shorthand:
//!                                            `f(&mut Sub, &str) -> Result<(), String>`
//! #[opt(skip)]                               not on any front-end (a computed field)
//! ```
//!
//! Struct tag: `#[options(krate = "crate")]` names the crate that holds
//! `options::KnobMeta` (`::lloom_tensor` by default; the tensor crate itself
//! says `crate`).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    Data, DeriveInput, Fields, GenericArgument, LitStr, PathArguments, Token, Type,
};

/// The parsed `#[opt(...)]` tag on one field.
#[derive(Default)]
struct OptAttr {
    flag: Option<String>,
    short: Option<char>,
    alias: Option<String>,
    env: Option<String>,
    variants: Vec<String>,
    parser: Option<String>,
    is_bool: bool,
    nested: bool,
    skip: bool,
}

impl Parse for OptAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut a = OptAttr::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "bool" => a.is_bool = true,
                "nested" => a.nested = true,
                "skip" => a.skip = true,
                "flag" => {
                    input.parse::<Token![=]>()?;
                    a.flag = Some(input.parse::<LitStr>()?.value());
                }
                "short" => {
                    input.parse::<Token![=]>()?;
                    a.short = Some(input.parse::<syn::LitChar>()?.value());
                }
                "alias" => {
                    input.parse::<Token![=]>()?;
                    a.alias = Some(input.parse::<LitStr>()?.value());
                }
                "env" => {
                    input.parse::<Token![=]>()?;
                    a.env = Some(input.parse::<LitStr>()?.value());
                }
                "parser" => {
                    input.parse::<Token![=]>()?;
                    a.parser = Some(input.parse::<LitStr>()?.value());
                }
                "variants" => {
                    input.parse::<Token![=]>()?;
                    let content;
                    syn::bracketed!(content in input);
                    let items = content.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
                    a.variants = items.into_iter().map(|s| s.value()).collect();
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[opt] key `{other}`"),
                    ))
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(a)
    }
}

/// The `#[options(...)]` tag on the struct.
struct OptionsAttr {
    krate: TokenStream,
}

impl Parse for OptionsAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut krate: TokenStream = quote!(::lloom_tensor);
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let v = input.parse::<LitStr>()?.value();
            match key.to_string().as_str() {
                "krate" => {
                    krate = v.parse().map_err(|_| {
                        syn::Error::new(key.span(), "krate must be a path like `crate`")
                    })?;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[options] key `{other}`"),
                    ))
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(OptionsAttr { krate })
    }
}

/// One field of the group, classified.
struct Field {
    ident: syn::Ident,
    ty: Type,
    attr: OptAttr,
    doc: String,
    /// `Option<T>`'s `T`, when the field is optional.
    inner: Option<Type>,
}

/// `Option<T>` → `Some(T)`.
fn option_inner(ty: &Type) -> Option<Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    }
}

/// The `///` lines of a field, joined.
fn doc_of(attrs: &[syn::Attribute]) -> String {
    let mut out = Vec::new();
    for a in attrs {
        if !a.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &a.meta {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
            {
                out.push(s.value().trim().to_string());
            }
        }
    }
    out.join(" ")
}

fn type_name(ty: &Type) -> String {
    quote!(#ty).to_string().replace(' ', "")
}

pub(crate) fn derive(input: DeriveInput) -> TokenStream {
    let struct_ident = &input.ident;
    let mut opts = OptionsAttr {
        krate: quote!(::lloom_tensor),
    };
    for a in &input.attrs {
        if a.path().is_ident("options") {
            match a.parse_args::<OptionsAttr>() {
                Ok(o) => opts = o,
                Err(e) => return e.to_compile_error(),
            }
        }
    }
    let krate = &opts.krate;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    struct_ident,
                    "#[derive(Options)] requires named fields",
                )
                .to_compile_error()
            }
        },
        _ => {
            return syn::Error::new_spanned(struct_ident, "#[derive(Options)] requires a struct")
                .to_compile_error()
        }
    };

    let mut items: Vec<Field> = Vec::new();
    for f in fields {
        let Some(ident) = f.ident.clone() else { continue };
        let mut attr = OptAttr::default();
        let mut tagged = false;
        for a in &f.attrs {
            if a.path().is_ident("opt") {
                tagged = true;
                // `#[opt]` with no arguments is a plain value field.
                if let syn::Meta::List(_) = &a.meta {
                    match a.parse_args::<OptAttr>() {
                        Ok(p) => attr = p,
                        Err(e) => return e.to_compile_error(),
                    }
                }
            }
        }
        if !tagged {
            attr.skip = true;
        }
        let inner = option_inner(&f.ty);
        items.push(Field {
            ident,
            ty: f.ty.clone(),
            attr,
            doc: doc_of(&f.attrs),
            inner,
        });
    }

    let args_ident = format_ident!("{}Args", struct_ident);

    // ---- 1. the clap companion --------------------------------------------
    let arg_fields = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        // clap's argument id defaults to the field name; two groups flattened
        // into one command may share one (`no_reuse`), so the id is qualified.
        let id = format!("{}.{}", struct_ident, ident);
        let flag = f
            .attr
            .flag
            .clone()
            .unwrap_or_else(|| ident.to_string().replace('_', "-"));
        let mut doc = f.doc.clone();
        if !f.attr.variants.is_empty() {
            doc.push_str(&format!(" [arms: {}]", f.attr.variants.join(", ")));
        }
        if let Some(env) = &f.attr.env {
            doc.push_str(&format!(" [env: {env}]"));
        }
        let short = match f.attr.short {
            Some(c) => quote! { short = #c, },
            None => quote! {},
        };
        let alias = match &f.attr.alias {
            Some(a) => quote! { alias = #a, },
            None => quote! {},
        };
        if f.attr.nested {
            let sub_args = nested_args_ty(&f.ty);
            // A nested group may also carry a compound shorthand
            // (`--readback ring-nonblocking` for `readback.{kind,eos}`): the
            // `parser` applies the spelling onto the group, then the group's
            // own flags override field by field.
            let compound = if f.attr.flag.is_some() && f.attr.parser.is_some() {
                let spelling = format_ident!("{}_spelling", ident);
                let flag = f.attr.flag.clone().unwrap();
                let doc = format!("{} (a shorthand for the group's keys)", f.doc);
                let sid = format!("{id}.spelling");
                quote! {
                    #[doc = #doc]
                    #[arg(id = #sid, long = #flag, value_name = "SPELLING")]
                    pub #spelling: Option<String>,
                }
            } else {
                quote! {}
            };
            quote! {
                #compound
                #[command(flatten)]
                pub #ident: #sub_args,
            }
        } else if f.attr.is_bool {
            quote! {
                #[doc = #doc]
                #[arg(id = #id, long = #flag, #short #alias num_args = 0..=1, default_missing_value = "true", action = ::clap::ArgAction::Set, value_parser = ::clap::value_parser!(bool), value_name = "BOOL")]
                pub #ident: Option<bool>,
            }
        } else {
            let value_name = type_name(f.inner.as_ref().unwrap_or(&f.ty)).to_uppercase();
            quote! {
                #[doc = #doc]
                #[arg(id = #id, long = #flag, #short #alias value_name = #value_name, allow_negative_numbers = true)]
                pub #ident: Option<String>,
            }
        }
    });

    let apply_stmts = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        if f.attr.nested {
            if f.attr.flag.is_some() && f.attr.parser.is_some() {
                let spelling = format_ident!("{}_spelling", ident);
                quote! {
                    if let Some(raw) = &self.#spelling {
                        base.set(#key, raw)?;
                    }
                    self.#ident.apply(&mut base.#ident)?;
                }
            } else {
                quote! { self.#ident.apply(&mut base.#ident)?; }
            }
        } else if f.attr.is_bool {
            quote! { if let Some(v) = self.#ident { base.#ident = v; } }
        } else {
            quote! {
                if let Some(raw) = &self.#ident {
                    base.set(#key, raw)?;
                }
            }
        }
    });

    // ---- 2. the dotted setter ---------------------------------------------
    let set_arms = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        if f.attr.nested {
            let whole = match &f.attr.parser {
                Some(parser) => {
                    let parse_fn: TokenStream = parser.parse().unwrap_or_else(|_| quote!(parse));
                    quote! { return #parse_fn(&mut self.#ident, value).map_err(|e| format!("{}: {}", #key, e)); }
                }
                None => quote! {
                    return Err(format!("`{}` is a group; name one of its keys (`{}.<key>`)", #key, #key));
                },
            };
            quote! {
                #key => {
                    let Some(rest) = rest else { #whole };
                    self.#ident.set(rest, value)
                }
            }
        } else if f.attr.is_bool {
            quote! {
                #key => {
                    if rest.is_some() { return Err(format!("`{}` takes no sub-key", #key)); }
                    self.#ident = #krate::options::parse_bool(value).map_err(|e| format!("{}: {}", #key, e))?;
                    Ok(())
                }
            }
        } else {
            let parse = parse_expr(f, krate);
            quote! {
                #key => {
                    if rest.is_some() { return Err(format!("`{}` takes no sub-key", #key)); }
                    self.#ident = #parse.map_err(|e| format!("{}: {}", #key, e))?;
                    Ok(())
                }
            }
        }
    });
    let key_list: Vec<String> = items
        .iter()
        .filter(|f| !f.attr.skip)
        .map(|f| f.ident.to_string())
        .collect();
    let keys_joined = key_list.join(", ");

    // ---- 3. merge_json ----------------------------------------------------
    let merge_arms = items.iter().map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        let ty = &f.ty;
        if f.attr.skip {
            // A field the group serializes but no front-end sets (computed,
            // or untagged): a dump read back must not trip on it.
            quote! { #key => {} }
        } else if f.attr.nested {
            // A string where the group is expected is its shorthand spelling.
            let by_string = match &f.attr.parser {
                Some(parser) => {
                    let parse_fn: TokenStream = parser.parse().unwrap_or_else(|_| quote!(parse));
                    quote! {
                        if let Some(s) = v.as_str() {
                            #parse_fn(&mut self.#ident, s).map_err(|e| format!("{}: {}", #key, e))?;
                            continue;
                        }
                    }
                }
                None => quote! {},
            };
            quote! { #key => { #by_string self.#ident.merge_json(v)?; } }
        } else {
            quote! {
                #key => {
                    self.#ident = ::serde_json::from_value::<#ty>(v.clone())
                        .map_err(|e| format!("{}: {}", #key, e))?;
                }
            }
        }
    });

    // ---- 4. from_env ------------------------------------------------------
    let env_stmts = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        if f.attr.nested {
            quote! { self.#ident.from_env()?; }
        } else if let Some(env) = &f.attr.env {
            quote! {
                if let Some(raw) = ::std::env::var_os(#env) {
                    let raw = raw.to_string_lossy().into_owned();
                    self.set(#key, &raw).map_err(|e| format!("{}: {}", #env, e))?;
                }
            }
        } else {
            quote! {}
        }
    });

    // ---- 5. META ----------------------------------------------------------
    let meta_items = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let key = f.ident.to_string();
        let flag = f
            .attr
            .flag
            .clone()
            .unwrap_or_else(|| key.replace('_', "-"));
        let doc = &f.doc;
        let env = match &f.attr.env {
            Some(e) => quote!(::std::option::Option::Some(#e)),
            None => quote!(::std::option::Option::None),
        };
        let variants = &f.attr.variants;
        let ty = type_name(&f.ty);
        if f.attr.nested {
            let sub = &f.ty;
            quote! {
                #krate::options::KnobMeta {
                    key: #key, flag: "", env: ::std::option::Option::None, doc: #doc,
                    variants: &[], ty: #ty, nested: ::std::option::Option::Some(<#sub>::META),
                    is_bool: false,
                }
            }
        } else {
            let is_bool = f.attr.is_bool;
            quote! {
                #krate::options::KnobMeta {
                    key: #key, flag: #flag, env: #env, doc: #doc,
                    variants: &[#(#variants),*], ty: #ty, nested: ::std::option::Option::None,
                    is_bool: #is_bool,
                }
            }
        }
    });

    let args_doc = format!(
        "clap flags for [`{struct_ident}`], generated by `#[derive(Options)]`: one `--<flag>` per \
         tagged field, nested groups flattened. A binary embeds it with `#[command(flatten)]` and \
         calls [`{args_ident}::apply`] onto a base built from the lower layers (defaults, the \
         checkpoint, the env aliases, a config file, a JSON group); a flag that was not given \
         leaves the base's value."
    );

    quote! {
        #[doc = #args_doc]
        #[derive(::clap::Args, Debug, Clone, Default)]
        pub struct #args_ident {
            #(#arg_fields)*
        }

        impl #args_ident {
            /// Write the flags that were given onto `base`; the rest keep the
            /// base's value. The error names the flag and the parse failure.
            pub fn apply(&self, base: &mut #struct_ident) -> ::std::result::Result<(), ::std::string::String> {
                #(#apply_stmts)*
                Ok(())
            }
        }

        impl #struct_ident {
            /// The group's keys, flags, env aliases, docs and arms — what
            /// `lloom knobs` prints and the drift test reads.
            pub const META: &'static [#krate::options::KnobMeta] = &[#(#meta_items),*];

            /// Set one knob by dotted key (`readback.eos`), the value in its
            /// flag spelling. Unknown keys and bad values are errors that name
            /// the key and, for a group, its keys.
            pub fn set(&mut self, key: &str, value: &str) -> ::std::result::Result<(), ::std::string::String> {
                let (head, rest) = match key.split_once('.') {
                    Some((h, r)) => (h, Some(r)),
                    None => (key, None),
                };
                match head {
                    #(#set_arms)*
                    other => Err(format!("unknown key `{}` (keys: {})", other, #keys_joined)),
                }
            }

            /// Merge a JSON object onto this group, field by field: a key the
            /// object does not name keeps its value; a nested group merges
            /// recursively; a value is deserialized into the field's type.
            pub fn merge_json(&mut self, json: &::serde_json::Value) -> ::std::result::Result<(), ::std::string::String> {
                let Some(map) = json.as_object() else {
                    return Err(format!("{}: expected a JSON object", stringify!(#struct_ident)));
                };
                for (k, v) in map {
                    match k.as_str() {
                        #(#merge_arms)*
                        other => return Err(format!("unknown key `{}` (keys: {})", other, #keys_joined)),
                    }
                }
                Ok(())
            }

            /// Read every env alias that is set onto this group (the layer that
            /// keeps `LLOOM_*` scripts working). A set but malformed value is an
            /// error naming the variable.
            pub fn from_env(&mut self) -> ::std::result::Result<(), ::std::string::String> {
                #(#env_stmts)*
                Ok(())
            }
        }
    }
}

/// `Foo` → `FooArgs` for a nested group's type.
fn nested_args_ty(ty: &Type) -> TokenStream {
    let Type::Path(p) = ty else {
        return quote!(#ty);
    };
    let mut p = p.clone();
    if let Some(last) = p.path.segments.last_mut() {
        last.ident = format_ident!("{}Args", last.ident);
    }
    quote!(#p)
}

/// The expression parsing `value: &str` into the field's type.
fn parse_expr(f: &Field, krate: &TokenStream) -> TokenStream {
    if let Some(parser) = &f.attr.parser {
        let parse_fn: TokenStream = parser.parse().unwrap_or_else(|_| quote!(parse));
        return quote!(#parse_fn(value));
    }
    match &f.inner {
        // `Option<T>`: `none` / `` clears; anything else parses as `T`.
        Some(inner) => quote! {
            (if value.is_empty() || value.eq_ignore_ascii_case("none") {
                Ok(::std::option::Option::None)
            } else {
                <#inner as ::std::str::FromStr>::from_str(value)
                    .map(::std::option::Option::Some)
                    .map_err(|e| e.to_string())
            })
        },
        None => {
            let ty = &f.ty;
            if type_name(ty) == "bool" {
                quote!(#krate::options::parse_bool(value))
            } else {
                quote!(<#ty as ::std::str::FromStr>::from_str(value).map_err(|e| e.to_string()))
            }
        }
    }
}
