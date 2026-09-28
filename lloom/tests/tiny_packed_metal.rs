//! P-c on Metal (`docs/porting-models.md` §21.8, INV-4): a synthesized packed
//! weight of every GGUF kind, multiplied on Metal, against the CPU's f32
//! product over the format's own `to_float` of the same bytes — at a decode
//! row (the matmul-vec arms) and at a prefill's rows (the matmul-mat arms),
//! with a row count that is not a multiple of any kernel's tile (§3.1b rule
//! 5). A kind Metal has no kernel for is refused by name, never wrong; every
//! refusal and every agreement is printed, and the refused set is pinned so
//! a kind that gains or loses its arm fails here.
#![cfg(feature = "metal")]

use lloom::testing::tiny::packed::{self, Synthesis};
use lloom::testing::tiny::{Role, ValueLaw};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Module, Shape};

#[test]
fn every_packed_kind_multiplies_on_metal_as_its_to_float_does() {
    let gpu = Device::new_metal(0).unwrap();
    let (n, k) = (37usize, 512usize);
    let role = Role::Matrix { fan_in: k };
    let mut refused = Vec::new();
    let mut bad = Vec::new();
    for kind in packed::ALL_KINDS
        .into_iter()
        .filter(|k| packed::is_packed(*k))
    {
        let shape = Shape::from((n, k));
        let name = format!("pc.{kind:?}");
        let w_cpu = packed::qtensor(
            ValueLaw::DEFAULT,
            3,
            &name,
            role,
            &shape,
            kind,
            Synthesis::Packed,
            &Device::Cpu,
        )
        .unwrap();
        let w_ref = w_cpu.dequantize(&Device::Cpu).unwrap();
        let w_gpu = packed::qtensor(
            ValueLaw::DEFAULT,
            3,
            &name,
            role,
            &shape,
            kind,
            Synthesis::Packed,
            &gpu,
        )
        .unwrap();
        let mm = QMatMul::from_qtensor(w_gpu).unwrap();
        for rows in [1usize, 7] {
            let x = ValueLaw::DEFAULT
                .tensor(
                    9,
                    "pc.x",
                    Role::Matrix { fan_in: 1 },
                    &Shape::from((1, rows, k)),
                    &Device::Cpu,
                )
                .unwrap();
            let want = x.broadcast_matmul(&w_ref.t().unwrap()).unwrap();
            let got = match mm.forward(&x.to_device(&gpu).unwrap()) {
                Ok(t) => t,
                Err(e) => {
                    let e = e.to_string();
                    eprintln!(
                        "{kind:?} rows {rows}: Metal refused: {}",
                        e.lines().next().unwrap_or("")
                    );
                    refused.push(format!("{kind:?}/{rows}"));
                    continue;
                }
            };
            let a: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = got
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            let finite = b.iter().all(|v| v.is_finite());
            eprintln!("{kind:?} rows {rows}: |metal - cpu| {d:e} at scale {scale:e}");
            // A misread block layout is order one of the product's scale;
            // accumulation order is ~1e-6 of it.
            if !finite || d > 1e-5 * scale {
                bad.push(format!(
                    "{kind:?} rows {rows}: {d:e} at scale {scale:e} (finite: {finite})"
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused, REFUSED, "the set of kinds Metal refuses moved");
}

/// The kinds (and row counts) Metal has no matmul arm for, as measured:
/// Q8_1 and Q8_K, the activation formats (their `kernel_mul_mv_*` is not
/// in the library, and the load fails by that name).
const REFUSED: &[&str] = &["Q8_1/1", "Q8_1/7", "Q8K/1", "Q8K/7"];
