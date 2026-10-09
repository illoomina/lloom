//! `#[derive(Options)]` — one option group, every front-end.
//!
//! An option group is a plain struct (`Default` + serde) whose fields are the
//! options a run selects. The derive emits, from the field list and the
//! `#[opt(...)]` tags, every way a value can reach it — so the flags, the
//! dotted keys, the JSON, the env names and the printed table are ONE source
//! of truth and cannot drift from each other:
//!
//! 1. `<Struct>Args`: a `clap::Args` companion (`--<flag>` per field, nested
//!    groups flattened), and `apply(&self, base: &mut Struct)` that writes
//!    only the flags that were given — "absent" keeps the base's value, so
//!    the layers compose (defaults < checkpoint < env < file < json < set <
//!    flags);
//! 2. `Struct::set(&mut self, key, value)`: a setter by storage path
//!    (`readback.eos = gpubatch`); a value in the key's vocabulary is
//!    converted to the stored one first;
//! 3. `Struct::merge_json(&mut self, &serde_json::Value)`: a JSON object in
//!    the storage shape, field by field so a partial object overrides only
//!    what it names;
//! 4. `Struct::from_env_with(&mut self, env)`: every legacy env name
//!    (`#[opt(env = "…")]`), then every key's canonical `LLOOM_<KEY>`;
//! 5. `Struct::META`: the static table (`KnobMeta`) the option index, the
//!    schema and `lloom options` read.
//!
//! Field tags:
//!
//! ```text
//! #[opt]                                     a value field: FromStr + serde
//! #[opt(flag = "kv-append")]                 the legacy long flag (default: the field name, kebab-case)
//! #[opt(short = 't', alias = "temp")]         a short flag, an alias for the long one
//! #[opt(bool)]                               a switch: `--x` / `--x=false` (`bool` or `Option<bool>`)
//! #[opt(env = "LLOOM_X")]                    the legacy env name
//! #[opt(variants = ["a", "b"])]              the arms, for help and the table
//! #[opt(parser = "parse_spark")]             `fn(&str) -> Result<T, String>` instead of FromStr
//! #[opt(nested)]                             a sub-group: its own derive, flattened
//! #[opt(nested, no_args)]                    …whose flags a front-end flattens itself
//! #[opt(nested, flag = "readback", parser = "f")]  …plus a compound shorthand:
//!                                            `f(&mut Sub, &str) -> Result<(), String>`
//! #[opt(skip)]                               not on any front-end (a computed field)
//!
//! The key (the option tree the user sees):
//! #[opt(path = "kernels.gemm.tile")]          the dotted key; the storage path is an alias
//! #[opt(status = "arm")]                     stable | advanced | arm | reference | instrument
//! #[opt(facets = ["wait"], flows = ["token-out"])]
//! #[opt(cli = "ctx-size", aliases = ["max-model-len"])]  a dedicated flag in --help;
//!                                            without one the legacy flag is hidden
//! #[opt(switches = ["cpu=cpu"])]              valueless flags writing a fixed value
//! #[opt(invert)]                             the key is the stored bool's negation
//! #[opt(values = ["event=true"])]             key spelling = stored spelling, where they differ
//! #[opt(from_v1 = "f", to_v1 = "g")]          the same, as functions of the spelling
//! #[opt(schema = "Wait", forms = ["spin", "sleep"])]  a shared value type and the forms this site takes
//! #[opt(legacy_only)]                        an alias of `path` that a write to the key resets
//! #[opt(legacy_keys = ["model.kv-append"])]  further accepted dotted keys
//! ```
//!
//! Struct tag: `#[options(krate = "crate")]` names the crate that holds
//! `options::KnobMeta` (`::lloom_tensor` by default; the tensor crate itself
//! says `crate`); `#[options(v0 = false)]` says the storage paths of this
//! group were never a spelling a user typed, so they are not aliases.
//!
//! A field's doc comment is its description. What `--help` and `lloom
//! options` print is that doc with its references (sections, issue numbers,
//! file names) taken out; the whole doc is kept for `lloom options
//! --internal`.

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
    no_args: bool,
    skip: bool,
    path: Option<String>,
    status: Option<String>,
    facets: Vec<String>,
    flows: Vec<String>,
    cli: Option<String>,
    aliases: Vec<String>,
    switches: Vec<(String, String)>,
    invert: bool,
    values: Vec<(String, String)>,
    from_v1: Option<String>,
    to_v1: Option<String>,
    schema: Option<String>,
    forms: Vec<String>,
    legacy_only: bool,
    legacy_keys: Vec<String>,
}

const STATUSES: [&str; 5] = ["stable", "advanced", "arm", "reference", "instrument"];

fn str_list(input: ParseStream) -> syn::Result<Vec<String>> {
    input.parse::<Token![=]>()?;
    let content;
    syn::bracketed!(content in input);
    let items = content.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
    Ok(items.into_iter().map(|s| s.value()).collect())
}

fn pair_list(input: ParseStream, what: &str) -> syn::Result<Vec<(String, String)>> {
    let span = input.span();
    str_list(input)?
        .into_iter()
        .map(|p| match p.split_once('=') {
            Some((a, b)) => Ok((a.to_string(), b.to_string())),
            None => Err(syn::Error::new(span, format!("{what}: `{p}` is not `a=b`"))),
        })
        .collect()
}

fn str_value(input: ParseStream) -> syn::Result<String> {
    input.parse::<Token![=]>()?;
    Ok(input.parse::<LitStr>()?.value())
}

impl Parse for OptAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut a = OptAttr::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "bool" => a.is_bool = true,
                "nested" => a.nested = true,
                "no_args" => a.no_args = true,
                "skip" => a.skip = true,
                "invert" => a.invert = true,
                "legacy_only" => a.legacy_only = true,
                "flag" => a.flag = Some(str_value(input)?),
                "short" => {
                    input.parse::<Token![=]>()?;
                    a.short = Some(input.parse::<syn::LitChar>()?.value());
                }
                "alias" => a.alias = Some(str_value(input)?),
                "env" => a.env = Some(str_value(input)?),
                "parser" => a.parser = Some(str_value(input)?),
                "variants" => a.variants = str_list(input)?,
                "path" => a.path = Some(str_value(input)?),
                "status" => {
                    let s = str_value(input)?;
                    if !STATUSES.contains(&s.as_str()) {
                        return Err(syn::Error::new(
                            key.span(),
                            format!("status `{s}` is not one of {}", STATUSES.join(", ")),
                        ));
                    }
                    a.status = Some(s);
                }
                "facets" => a.facets = str_list(input)?,
                "flows" => a.flows = str_list(input)?,
                "cli" => a.cli = Some(str_value(input)?),
                "aliases" => a.aliases = str_list(input)?,
                "switches" => a.switches = pair_list(input, "switches")?,
                "values" => a.values = pair_list(input, "values")?,
                "from_v1" => a.from_v1 = Some(str_value(input)?),
                "to_v1" => a.to_v1 = Some(str_value(input)?),
                "schema" => a.schema = Some(str_value(input)?),
                "forms" => a.forms = str_list(input)?,
                "legacy_keys" => a.legacy_keys = str_list(input)?,
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
    v0: bool,
}

impl Parse for OptionsAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut krate: TokenStream = quote!(::lloom_tensor);
        let mut v0 = true;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "krate" => {
                    let v = input.parse::<LitStr>()?.value();
                    krate = v.parse().map_err(|_| {
                        syn::Error::new(key.span(), "krate must be a path like `crate`")
                    })?;
                }
                "v0" => v0 = input.parse::<syn::LitBool>()?.value,
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
        Ok(OptionsAttr { krate, v0 })
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

impl Field {
    /// `Option<bool>`, the tri-state switch (unset = the front-end's default).
    fn optional_bool(&self) -> bool {
        self.inner.as_ref().map(type_name).as_deref() == Some("bool")
    }
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

/// Whether `s` names a section, an issue, a measured probe or a source file
/// — what a user-facing description leaves out.
pub(crate) fn is_ref(s: &str) -> bool {
    if s.contains('§') || s.contains("DESIGN") || s.contains("docs/") {
        return true;
    }
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'#' && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
            return true;
        }
    }
    s.split(|c: char| c.is_whitespace() || "(),;`'\"".contains(c))
        .any(ref_word)
}

/// One word that is a reference: `foo.rs`, `bar.md`, `x.metal`, `P31`.
fn ref_word(w: &str) -> bool {
    let w = w.trim_end_matches(['.', ':']);
    for ext in [".rs", ".md", ".metal", ".sh", ".py"] {
        if w.len() > ext.len() && w.ends_with(ext) {
            return true;
        }
    }
    let mut c = w.chars();
    c.next() == Some('P') && {
        let rest: String = c.collect();
        !rest.is_empty() && rest.chars().all(|d| d.is_ascii_digit())
    }
}

/// The description without its references: parenthesized asides that name
/// one are dropped whole, and the reference words left in running text are
/// dropped with the punctuation they leave dangling.
pub(crate) fn user_doc(doc: &str) -> String {
    // 1. parentheses (balanced) whose content is a reference.
    let mut out = String::new();
    let chars: Vec<char> = doc.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            let mut depth = 0;
            let mut j = i;
            while j < chars.len() {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if j < chars.len() {
                let inner: String = chars[i + 1..j].iter().collect();
                if is_ref(&inner) {
                    // drop the space before the aside too
                    while out.ends_with(' ') {
                        out.pop();
                    }
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    // 2. reference words in running text.
    let words: Vec<String> = out
        .split(' ')
        .filter(|w| {
            let bare = w.trim_matches(|c: char| ",;.:`'\"".contains(c));
            !(bare.starts_with('§')
                || (bare.starts_with('#') && bare[1..].starts_with(|c: char| c.is_ascii_digit()))
                || bare.starts_with("pre-#")
                || ref_word(bare))
        })
        .map(str::to_string)
        .collect();
    let mut s = words.join(" ");
    // 3. the punctuation the removals leave behind.
    for (from, to) in [
        ("  ", " "),
        (" ,", ","),
        (" .", "."),
        (" ;", ";"),
        (" :", ":"),
        (" )", ")"),
        ("( ", "("),
        ("()", ""),
        (",.", "."),
        (";.", "."),
        (", ,", ","),
        ("issue .", "."),
        ("the issue's", "its"),
    ] {
        while s.contains(from) {
            s = s.replace(from, to);
        }
    }
    s.trim().trim_end_matches([',', ';']).to_string()
}

fn status_tokens(krate: &TokenStream, s: Option<&str>) -> TokenStream {
    match s.unwrap_or("advanced") {
        "stable" => quote!(#krate::options::Status::Stable),
        "arm" => quote!(#krate::options::Status::Arm),
        "reference" => quote!(#krate::options::Status::Reference),
        "instrument" => quote!(#krate::options::Status::Instrument),
        _ => quote!(#krate::options::Status::Advanced),
    }
}

fn path_tokens(s: &Option<String>) -> TokenStream {
    match s {
        Some(p) => {
            let p: TokenStream = p.parse().unwrap_or_else(|_| quote!(parse));
            quote!(#p)
        }
        None => quote!(),
    }
}

fn canonical_env(path: &str) -> String {
    format!("LLOOM_{}", path.to_ascii_uppercase().replace('.', "__"))
}

pub(crate) fn derive(input: DeriveInput) -> TokenStream {
    let struct_ident = &input.ident;
    let mut opts = OptionsAttr {
        krate: quote!(::lloom_tensor),
        v0: true,
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
    let v0 = opts.v0;

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
        if attr.invert && !attr.is_bool {
            return syn::Error::new_spanned(&f.ty, "#[opt(invert)] needs a bool field")
                .to_compile_error();
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
    let arg_fields = items
        .iter()
        .filter(|f| !(f.attr.skip || f.attr.nested && f.attr.no_args))
        .map(|f| {
            let ident = &f.ident;
            // clap's argument id defaults to the field name; two groups flattened
            // into one command may share one (`no_reuse`), so the id is qualified.
            let id = format!("{}.{}", struct_ident, ident);
            let legacy = f
                .attr
                .flag
                .clone()
                .unwrap_or_else(|| ident.to_string().replace('_', "-"));
            // The dedicated flag is shown; a key without one keeps its legacy
            // flag, hidden (`lloom options` lists it). A group without keys
            // keeps the old behaviour: every flag shown.
            let (long, mut aliases) = match &f.attr.cli {
                Some(cli) => (
                    cli.clone(),
                    if *cli != legacy { vec![legacy.clone()] } else { vec![] },
                ),
                None => (legacy.clone(), vec![]),
            };
            aliases.extend(f.attr.aliases.iter().cloned());
            if let Some(a) = &f.attr.alias {
                aliases.push(a.clone());
            }
            let hidden = f.attr.path.is_some() && f.attr.cli.is_none();
            let heading = match &f.attr.path {
                Some(p) => quote! { help_heading = #krate::options::heading(#p), },
                None => quote! {},
            };
            let mut doc = user_doc(&f.doc);
            if !f.attr.variants.is_empty() {
                doc.push_str(&format!(" [arms: {}]", f.attr.variants.join(", ")));
            }
            if let Some(p) = &f.attr.path {
                doc.push_str(&format!(" [key: {p}]"));
            } else if let Some(env) = &f.attr.env {
                doc.push_str(&format!(" [env: {env}]"));
            }
            let short = match f.attr.short {
                Some(c) => quote! { short = #c, },
                None => quote! {},
            };
            let alias_attr = if aliases.is_empty() {
                quote! {}
            } else {
                quote! { aliases = [#(#aliases),*], }
            };
            if f.attr.nested {
                let sub_args = nested_args_ty(&f.ty);
                // A nested group may also carry a compound shorthand
                // (`--readback ring-nonblocking` for `readback.{kind,eos}`): the
                // `parser` applies the spelling onto the group, then the group's
                // own flags override field by field.
                let compound = if f.attr.flag.is_some() && f.attr.parser.is_some() {
                    let spelling = format_ident!("{}_spelling", ident);
                    let doc = match &f.attr.path {
                        Some(p) => format!("{} [key: {p}]", user_doc(&f.doc)),
                        None => user_doc(&f.doc),
                    };
                    let sid = format!("{id}.spelling");
                    quote! {
                        #[doc = #doc]
                        #[arg(id = #sid, long = #long, #alias_attr value_name = "SPELLING", hide = #hidden, #heading overrides_with = #sid)]
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
            } else {
                let main = if f.attr.is_bool {
                    quote! {
                        #[doc = #doc]
                        #[arg(id = #id, long = #long, #short #alias_attr num_args = 0..=1, default_missing_value = "true", action = ::clap::ArgAction::Set, value_parser = #krate::options::parse_bool, value_name = "BOOL", hide = #hidden, #heading overrides_with = #id)]
                        pub #ident: Option<bool>,
                    }
                } else {
                    let value_name = value_name(f);
                    quote! {
                        #[doc = #doc]
                        #[arg(id = #id, long = #long, #short #alias_attr value_name = #value_name, allow_negative_numbers = true, hide = #hidden, #heading overrides_with = #id)]
                        pub #ident: Option<String>,
                    }
                };
                let switches = f.attr.switches.iter().enumerate().map(|(i, (flag, value))| {
                    let sw = format_ident!("{}_switch{}", ident, i);
                    let sid = format!("{id}.{flag}");
                    let sdoc = format!("Same as `--{long} {value}`.");
                    quote! {
                        #[doc = #sdoc]
                        #[arg(id = #sid, long = #flag, action = ::clap::ArgAction::SetTrue, hide = #hidden, #heading conflicts_with = #id)]
                        pub #sw: bool,
                    }
                });
                quote! { #main #(#switches)* }
            }
        });

    let apply_stmts = items
        .iter()
        .filter(|f| !(f.attr.skip || f.attr.nested && f.attr.no_args))
        .map(|f| {
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
            } else {
                let main = if f.attr.is_bool {
                    quote! {
                        if let Some(v) = self.#ident {
                            base.set(#key, if v { "true" } else { "false" })?;
                        }
                    }
                } else {
                    quote! {
                        if let Some(raw) = &self.#ident {
                            base.set(#key, raw)?;
                        }
                    }
                };
                let switches = f.attr.switches.iter().enumerate().map(|(i, (_, value))| {
                    let sw = format_ident!("{}_switch{}", ident, i);
                    quote! { if self.#sw { base.set(#key, #value)?; } }
                });
                quote! { #main #(#switches)* }
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
            return quote! {
                #key => {
                    let Some(rest) = rest else { #whole };
                    self.#ident.set(rest, value)
                }
            };
        }
        // The key's vocabulary into the stored one, before the field's parser.
        let value_map = f.attr.values.iter().map(|(k, s)| quote! { #k => #s.to_string(), });
        let from_v1 = match &f.attr.from_v1 {
            Some(p) => {
                let p = path_tokens(&Some(p.clone()));
                quote! { let value = #p(&value).map_err(|e| format!("{}: {}", #key, e))?; }
            }
            None => quote! {},
        };
        let convert = quote! {
            let value: ::std::string::String = match value {
                #(#value_map)*
                other => other.to_string(),
            };
            #from_v1
            let value: &str = &value;
        };
        if f.attr.is_bool {
            let assign = if f.optional_bool() {
                quote! {
                    self.#ident = if #krate::options::is_none(value) {
                        ::std::option::Option::None
                    } else {
                        ::std::option::Option::Some(#krate::options::parse_bool(value).map_err(|e| format!("{}: {}", #key, e))?)
                    };
                }
            } else {
                quote! {
                    self.#ident = #krate::options::parse_bool(value).map_err(|e| format!("{}: {}", #key, e))?;
                }
            };
            quote! {
                #key => {
                    if rest.is_some() { return Err(format!("`{}` takes no sub-key", #key)); }
                    #convert
                    #assign
                    Ok(())
                }
            }
        } else {
            let parse = parse_expr(f, krate);
            quote! {
                #key => {
                    if rest.is_some() { return Err(format!("`{}` takes no sub-key", #key)); }
                    #convert
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

    // ---- 4. the env -------------------------------------------------------
    let env_legacy = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        if f.attr.nested {
            quote! { self.#ident.env_legacy(env)?; }
        } else if let Some(var) = &f.attr.env {
            quote! {
                if let Some(raw) = env(#var) {
                    self.set(#key, &raw).map_err(|e| format!("{}: {}", #var, e))?;
                }
            }
        } else {
            quote! {}
        }
    });
    let env_canonical = items.iter().filter(|f| !f.attr.skip).map(|f| {
        let ident = &f.ident;
        let key = ident.to_string();
        if f.attr.nested {
            return quote! { self.#ident.env_canonical(env)?; };
        }
        let Some(path) = &f.attr.path else {
            return quote! {};
        };
        let var = canonical_env(path);
        if f.attr.legacy_only {
            // The key names another field; setting it makes this alias's
            // value moot, so it returns to its default.
            return quote! {
                if env(#var).is_some() {
                    self.#ident = <Self as ::std::default::Default>::default().#ident;
                }
            };
        }
        let invert = if f.attr.invert {
            quote! {
                let raw = (!#krate::options::parse_bool(&raw).map_err(|e| format!("{}: {}", #var, e))?).to_string();
            }
        } else {
            quote! {}
        };
        quote! {
            if let Some(raw) = env(#var) {
                #invert
                self.set(#key, &raw).map_err(|e| format!("{}: {}", #var, e))?;
            }
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
        let internal = &f.doc;
        let doc = user_doc(&f.doc);
        let env = match &f.attr.env {
            Some(e) => quote!(::std::option::Option::Some(#e)),
            None => quote!(::std::option::Option::None),
        };
        let variants = &f.attr.variants;
        let ty = type_name(&f.ty);
        let path = f.attr.path.clone().unwrap_or_default();
        let status = status_tokens(krate, f.attr.status.as_deref());
        let facets = &f.attr.facets;
        let flows = &f.attr.flows;
        let cli = f.attr.cli.clone().unwrap_or_default();
        let short = match f.attr.short {
            Some(c) => quote!(::std::option::Option::Some(#c)),
            None => quote!(::std::option::Option::None),
        };
        let mut aliases = f.attr.aliases.clone();
        if let Some(a) = &f.attr.alias {
            aliases.push(a.clone());
        }
        let sw_flags = f.attr.switches.iter().map(|(a, _)| a);
        let sw_values = f.attr.switches.iter().map(|(_, b)| b);
        let invert = f.attr.invert;
        let val_k = f.attr.values.iter().map(|(a, _)| a);
        let val_s = f.attr.values.iter().map(|(_, b)| b);
        let from_v1 = match &f.attr.from_v1 {
            Some(p) => {
                let p = path_tokens(&Some(p.clone()));
                quote!(::std::option::Option::Some(#p as #krate::options::FromKey))
            }
            None => quote!(::std::option::Option::None),
        };
        let to_v1 = match &f.attr.to_v1 {
            Some(p) => {
                let p = path_tokens(&Some(p.clone()));
                quote!(::std::option::Option::Some(#p as fn(&str) -> ::std::string::String))
            }
            None => quote!(::std::option::Option::None),
        };
        let schema = f.attr.schema.clone().unwrap_or_default();
        let forms = &f.attr.forms;
        let legacy_only = f.attr.legacy_only;
        let legacy_keys = &f.attr.legacy_keys;
        let compound = f.attr.nested && f.attr.flag.is_some() && f.attr.parser.is_some();
        let nested = if f.attr.nested {
            let sub = &f.ty;
            quote!(::std::option::Option::Some(<#sub>::META))
        } else {
            quote!(::std::option::Option::None)
        };
        let flag = if f.attr.nested && !compound { String::new() } else { flag };
        let is_bool = f.attr.is_bool;
        quote! {
            #krate::options::KnobMeta {
                key: #key, flag: #flag, env: #env, doc: #doc,
                variants: &[#(#variants),*], ty: #ty, nested: #nested,
                is_bool: #is_bool,
                path: #path, status: #status,
                facets: &[#(#facets),*], flows: &[#(#flows),*],
                cli: #cli, short: #short, aliases: &[#(#aliases),*],
                switches: &[#((#sw_flags, #sw_values)),*],
                invert: #invert,
                values: &[#((#val_k, #val_s)),*],
                from_v1: #from_v1, to_v1: #to_v1,
                schema: #schema, forms: &[#(#forms),*],
                internal: #internal,
                legacy_only: #legacy_only,
                v0: #v0,
                legacy_keys: &[#(#legacy_keys),*],
                compound: #compound,
            }
        }
    });

    let args_doc = format!(
        "clap flags for [`{struct_ident}`], generated by `#[derive(Options)]`: one `--<flag>` per \
         tagged field, nested groups flattened. A binary embeds it with `#[command(flatten)]` and \
         calls [`{args_ident}::apply`] onto a base built from the lower layers (defaults, the \
         checkpoint, the env, a config file, a JSON group); a flag that was not given leaves the \
         base's value."
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
            /// The group's options: storage segment, key, flags, env names,
            /// docs, arms — what the option index, the schema and `lloom
            /// options` read.
            pub const META: &'static [#krate::options::KnobMeta] = &[#(#meta_items),*];

            /// Set one option by storage path (`readback.eos`), the value in
            /// its flag spelling or its key's. Unknown keys and bad values are
            /// errors that name the key and, for a group, its keys.
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

            /// Merge a JSON object in the storage shape onto this group, field
            /// by field: a key the object does not name keeps its value; a
            /// nested group merges recursively; a value is deserialized into
            /// the field's type.
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

            /// Read the process environment ([`Self::from_env_with`]).
            pub fn from_env(&mut self) -> ::std::result::Result<(), ::std::string::String> {
                self.from_env_with(&|name: &str| {
                    ::std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
                })
            }

            /// Read every env name `env` resolves: the legacy names first,
            /// then each key's `LLOOM_<KEY>`, which wins. A set but malformed
            /// value is an error naming the variable.
            pub fn from_env_with(&mut self, env: &dyn Fn(&str) -> ::std::option::Option<::std::string::String>) -> ::std::result::Result<(), ::std::string::String> {
                self.env_legacy(env)?;
                self.env_canonical(env)
            }

            #[doc(hidden)]
            pub fn env_legacy(&mut self, env: &dyn Fn(&str) -> ::std::option::Option<::std::string::String>) -> ::std::result::Result<(), ::std::string::String> {
                let _ = env;
                #(#env_legacy)*
                Ok(())
            }

            #[doc(hidden)]
            pub fn env_canonical(&mut self, env: &dyn Fn(&str) -> ::std::option::Option<::std::string::String>) -> ::std::result::Result<(), ::std::string::String> {
                let _ = env;
                #(#env_canonical)*
                Ok(())
            }
        }
    }
}

/// The placeholder `--help` shows for a field's value.
fn value_name(f: &Field) -> String {
    if !f.attr.variants.is_empty() {
        return "ARM".into();
    }
    let ty = type_name(f.inner.as_ref().unwrap_or(&f.ty));
    match ty.rsplit("::").next().unwrap_or(&ty) {
        "PathBuf" => "PATH",
        "usize" | "u64" | "u32" | "u16" | "u8" => "N",
        "f64" | "f32" => "X",
        _ => "VALUE",
    }
    .into()
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

/// The expression parsing `value: &str` into the field's type, through the
/// value grammar (`lloom_tensor::options`): a bool, a count and a number
/// parse there; any other type through its `FromStr`, which owns its own
/// arms. An optional typed value is unset by `none` and refuses the empty
/// value; an optional text (a path, a name) is unset by `none` or the empty
/// value, which is no text.
fn parse_expr(f: &Field, krate: &TokenStream) -> TokenStream {
    if let Some(parser) = &f.attr.parser {
        let parse_fn: TokenStream = parser.parse().unwrap_or_else(|_| quote!(parse));
        return quote!(#parse_fn(value));
    }
    let typed = |ty: &Type| -> Option<TokenStream> {
        match type_name(ty).as_str() {
            "bool" => Some(quote!(#krate::options::parse_bool(value))),
            "usize" | "u64" | "u32" | "u16" | "u8" => {
                Some(quote!(#krate::options::parse_count::<#ty>(value)))
            }
            "f64" => Some(quote!(#krate::options::parse_number(value))),
            _ => None,
        }
    };
    match &f.inner {
        Some(inner) => match typed(inner) {
            Some(parse) => quote! {
                (if #krate::options::is_none(value) {
                    Ok(::std::option::Option::None)
                } else {
                    #parse.map(::std::option::Option::Some)
                })
            },
            None => quote! {
                (if value.trim().is_empty() || #krate::options::is_none(value) {
                    Ok(::std::option::Option::None)
                } else {
                    <#inner as ::std::str::FromStr>::from_str(value)
                        .map(::std::option::Option::Some)
                        .map_err(|e| e.to_string())
                })
            },
        },
        None => {
            let ty = &f.ty;
            let text = matches!(
                type_name(ty).as_str(),
                "String" | "PathBuf" | "std::path::PathBuf"
            );
            typed(ty).unwrap_or_else(|| {
                if text {
                    quote!(<#ty as ::std::str::FromStr>::from_str(value).map_err(|e| e.to_string()))
                } else {
                    // An enum's own `FromStr`: its arm in any case, the empty
                    // value refused.
                    quote! {
                        #krate::options::non_empty(value).and_then(|v| {
                            <#ty as ::std::str::FromStr>::from_str(&v.to_ascii_lowercase())
                                .map_err(|e| e.to_string())
                        })
                    }
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{is_ref, user_doc};

    #[test]
    fn a_reference_is_taken_out_of_the_description() {
        for (doc, want) in [
            (
                "The `m` up to which the skinny tile is taken; above it the 64-row tile (§13.5o/p).",
                "The `m` up to which the skinny tile is taken; above it the 64-row tile.",
            ),
            (
                "`mlock` the uploads (the slot pools' fix applied to the dense weights; #749 P29).",
                "`mlock` the uploads.",
            ),
            (
                "Shadow every host upload and check it back (the #748 corruption instrument).",
                "Shadow every host upload and check it back.",
            ),
            (
                "Lift the quarantine on the kernels (the #732 reproducer's switch).",
                "Lift the quarantine on the kernels.",
            ),
            (
                "The ICB replay executor (#146/#169): the stable subset recorded.",
                "The ICB replay executor: the stable subset recorded.",
            ),
            ("Routed GEMM (`quantized/metal.rs`) with no aside.", "Routed GEMM with no aside."),
        ] {
            let got = user_doc(doc);
            assert_eq!(got, want, "{doc}");
            assert!(!is_ref(&got), "{got}");
        }
        assert!(
            is_ref("see P31 (4)") && is_ref("`gemv.metal`") && !is_ref("generation_config.json")
        );
    }
}
