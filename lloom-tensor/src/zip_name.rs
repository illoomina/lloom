//! Archive-key compatibility across zip dependency versions.
//!
//! zip8 honored the ZIP language flag; zip9 tries UTF-8 first even when the
//! bytes are flagged CP437. These are tensor keys, not filesystem paths.
//! Leave archive parsing/Unicode-extra validation to zip, and decode its
//! resulting raw name/flags with the established language contract.

use std::borrow::Cow;

pub(crate) fn decoded(raw: &[u8], flags: u16) -> Cow<'_, str> {
    if flags & 0x0800 != 0 || raw.is_ascii() {
        return String::from_utf8_lossy(raw);
    }
    Cow::Owned(
        raw.iter()
            .map(|&b| {
                if b < 128 {
                    char::from(b)
                } else {
                    CP437_HIGH[usize::from(b - 128)]
                }
            })
            .collect(),
    )
}

// IBM CP437 high half. Matches the legacy zip8 decoder and Unicode's CP437
// mapping; low half is ASCII. No archive parsing or path interpretation here.
const CP437_HIGH: [char; 128] = [
    'Ç', 'ü', 'é', 'â', 'ä', 'à', 'å', 'ç', 'ê', 'ë', 'è', 'ï', 'î', 'ì', 'Ä', 'Å', 'É', 'æ', 'Æ',
    'ô', 'ö', 'ò', 'û', 'ù', 'ÿ', 'Ö', 'Ü', '¢', '£', '¥', '₧', 'ƒ', 'á', 'í', 'ó', 'ú', 'ñ', 'Ñ',
    'ª', 'º', '¿', '⌐', '¬', '½', '¼', '¡', '«', '»', '░', '▒', '▓', '│', '┤', '╡', '╢', '╖', '╕',
    '╣', '║', '╗', '╝', '╜', '╛', '┐', '└', '┴', '┬', '├', '─', '┼', '╞', '╟', '╚', '╔', '╩', '╦',
    '╠', '═', '╬', '╧', '╨', '╤', '╥', '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌', '█', '▄', '▌', '▐',
    '▀', 'α', 'ß', 'Γ', 'π', 'Σ', 'σ', 'µ', 'τ', 'Φ', 'Θ', 'Ω', 'δ', '∞', 'φ', 'ε', '∩', '≡', '±',
    '≥', '≤', '⌠', '⌡', '÷', '≈', '°', '∙', '·', '√', 'ⁿ', '²', '■', ' ',
];
