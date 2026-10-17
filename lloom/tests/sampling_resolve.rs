//! The sampling contract's edges at its one entry point, `Sampling::resolve`
//! (`design/docs/dispatch.md` §11.2j): every row of the truth table, the arm
//! it resolves to or the field it refuses by name.

use lloom::generation::{Sampling, GREEDY_BELOW};

#[derive(Debug)]
enum Want {
    Arm(Sampling),
    /// Refused, the message naming this field.
    Refused(&'static str),
}

fn arm(s: Sampling) -> Want {
    Want::Arm(s)
}

const T: f64 = 0.7;

/// `(temperature, top_k, top_p, gumbel)` and what they resolve to.
#[allow(clippy::type_complexity)]
fn table() -> Vec<((Option<f64>, Option<usize>, Option<f64>, bool), Want)> {
    use Sampling::*;
    vec![
        // Temperature: unset, zero and below the threshold are greedy; a
        // value that is not a finite number >= 0 is refused.
        ((None, None, None, false), arm(ArgMax)),
        ((Some(0.0), None, None, false), arm(ArgMax)),
        ((Some(-0.0), None, None, false), arm(ArgMax)),
        ((Some(GREEDY_BELOW / 2.0), None, None, false), arm(ArgMax)),
        (
            (Some(GREEDY_BELOW), None, None, false),
            arm(All {
                temperature: GREEDY_BELOW,
            }),
        ),
        ((Some(T), None, None, false), arm(All { temperature: T })),
        (
            (Some(-0.5), None, None, false),
            Want::Refused("temperature"),
        ),
        (
            (Some(f64::NAN), None, None, false),
            Want::Refused("temperature"),
        ),
        (
            (Some(f64::INFINITY), None, None, false),
            Want::Refused("temperature"),
        ),
        // Greedy validates the fields it does not read, and ignores the
        // valid ones (and the method).
        ((None, Some(50), Some(0.9), false), arm(ArgMax)),
        ((Some(0.0), Some(20), Some(0.95), true), arm(ArgMax)),
        ((None, None, Some(1.5), false), Want::Refused("top_p")),
        (
            (Some(0.0), None, Some(f64::NAN), false),
            Want::Refused("top_p"),
        ),
        // top_k: 0 is off; any positive k is the filter's to judge against
        // the row (k >= V is off there).
        ((Some(T), Some(0), None, false), arm(All { temperature: T })),
        (
            (Some(T), Some(1), None, false),
            arm(TopK {
                k: 1,
                temperature: T,
            }),
        ),
        (
            (Some(T), Some(50), None, false),
            arm(TopK {
                k: 50,
                temperature: T,
            }),
        ),
        (
            (Some(T), Some(usize::MAX), None, false),
            arm(TopK {
                k: usize::MAX,
                temperature: T,
            }),
        ),
        // top_p: 1 is off, 0 keeps the top candidate, out of [0, 1] refused.
        (
            (Some(T), None, Some(1.0), false),
            arm(All { temperature: T }),
        ),
        (
            (Some(T), None, Some(0.0), false),
            arm(TopP {
                p: 0.0,
                temperature: T,
            }),
        ),
        (
            (Some(T), None, Some(0.95), false),
            arm(TopP {
                p: 0.95,
                temperature: T,
            }),
        ),
        (
            (Some(T), None, Some(1.0 + 1e-9), false),
            Want::Refused("top_p"),
        ),
        (
            (Some(T), None, Some(-1e-300), false),
            Want::Refused("top_p"),
        ),
        (
            (Some(T), None, Some(f64::NAN), false),
            Want::Refused("top_p"),
        ),
        (
            (Some(T), None, Some(f64::INFINITY), false),
            Want::Refused("top_p"),
        ),
        // Both.
        (
            (Some(T), Some(20), Some(0.95), false),
            arm(TopKThenTopP {
                k: 20,
                p: 0.95,
                temperature: T,
            }),
        ),
        (
            (Some(T), Some(20), Some(1.0), false),
            arm(TopK {
                k: 20,
                temperature: T,
            }),
        ),
        (
            (Some(T), Some(0), Some(0.5), false),
            arm(TopP {
                p: 0.5,
                temperature: T,
            }),
        ),
        (
            (Some(T), Some(0), Some(1.0), false),
            arm(All { temperature: T }),
        ),
        // Filtered Gumbel preserves the method and resolves to the explicit host arm.
        (
            (Some(T), None, None, true),
            arm(GumbelSoftmax { temperature: T }),
        ),
        (
            (Some(T), Some(0), Some(1.0), true),
            arm(GumbelSoftmax { temperature: T }),
        ),
        ((Some(T), Some(20), None, true), arm(FilteredGumbel { temperature: T, k: 20, p: 1.0 })),
        ((Some(T), None, Some(0.9), true), arm(FilteredGumbel { temperature: T, k: 0, p: 0.9 })),
        (
            (Some(T), Some(20), Some(0.95), true),
            arm(FilteredGumbel { temperature: T, k: 20, p: 0.95 }),
        ),
    ]
}

#[test]
fn every_row_of_the_truth_table_resolves_as_written() {
    let rows = table();
    let (mut arms, mut refusals) = (0, 0);
    for ((t, k, p, g), want) in &rows {
        let got = Sampling::resolve(*t, *k, *p, *g);
        match (want, got) {
            (Want::Arm(w), Ok(s)) => {
                assert_eq!(&s, w, "{t:?} {k:?} {p:?} gumbel={g}");
                arms += 1;
            }
            (Want::Refused(field), Err(e)) => {
                let e = e.to_string();
                assert!(
                    e.contains(field),
                    "{t:?} {k:?} {p:?} gumbel={g}: refused without naming {field}: {e}"
                );
                refusals += 1;
            }
            (w, g2) => panic!("{t:?} {k:?} {p:?} gumbel={g}: want {w:?}, got {g2:?}"),
        }
    }
    println!(
        "RESOLVE rows={} arms={arms} refusals={refusals}",
        rows.len()
    );
    assert_eq!((arms, refusals), (24, 9));
}
