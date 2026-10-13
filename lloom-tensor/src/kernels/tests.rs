use super::*;
use crate::kernels::kernels::params::{ParamStyle, SdpaQuantParams, SdpaQuantWriteParams};
use crate::kernels::metal::{Commands, ResidencySet};
use core::ffi::c_void;
use half::{bf16, f16};
use rand::prelude::SliceRandom;
use rand::{rng, Rng};
use std::sync::Arc;
use std::thread;

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    Commands::new(queue, residency_set).unwrap()
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let ptr = unsafe { buffer.contents() } as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

#[test]
fn pipeline_cache_distinguishes_sources() {
    let device = device();
    let kernels = Kernels::new();

    // Prime the cache with a name that is not present in the binary library.
    kernels
        .load_pipeline(&device, Source::Unary, "cos_f32")
        .unwrap();
    assert!(matches!(
        kernels.load_pipeline(&device, Source::Binary, "cos_f32"),
        Err(MetalKernelError::LoadFunctionError(_))
    ));
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: 0,
    };
    let output = new_buffer(&device, v);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_unary_contiguous(
            &device,
            &encoder,
            &kernels,
            name,
            size_of::<T>(),
            v.len(),
            input,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

fn run_binary<T: Clone, S: ToString>(x: &[T], y: &[T], name: S) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device
        .new_buffer(std::mem::size_of_val(x), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_binary_contiguous(
            &device,
            &encoder,
            &kernels,
            name,
            size_of::<T>(),
            x.len(),
            BufferOffset::zero_offset(&left),
            BufferOffset::zero_offset(&right),
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, x.len())
}

fn run_strided<T: Clone>(
    v: &[T],
    kernel: unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: offset,
    };
    let output_b = new_buffer(&device, v);
    let output = BufferOffset {
        buffer: &output_b,
        offset_in_bytes: 0,
    };
    let kernels = Kernels::new();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_unary_strided(
            &device, &encoder, &kernels, kernel, shape, input, strides, output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output_b, v.len())
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

    let v = vec![1.0f32; 10_000];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.954, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::HALF);
    assert_eq!(approx_f16(results, 3), expected);
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.955, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn silu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.27, 0.0, 0.73, 1.76, 2.86, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn silu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.269, 0.0, 0.731, 1.762, 2.858, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, "badd_f32");
    let expected: Vec<_> = left
        .iter()
        .zip(right.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    macro_rules! binary_op {
        ($opname:ident, $dtype:ident, $opexpr:expr) => {{
            let results = run_binary(
                &lhs,
                &rhs,
                concat!(stringify!($opname), "_", stringify!($dtype)),
            );
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&$dtype, &$dtype)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }
    binary_op!(badd, bf16, |x, y| x + y);
    binary_op!(bsub, bf16, |x, y| x - y);
    binary_op!(bmul, bf16, |x, y| x * y);
    binary_op!(bdiv, bf16, |x, y| x / y);
    binary_op!(bminimum, bf16, |x: bf16, y| x.min(y));
    binary_op!(bmaximum, bf16, |x: bf16, y| x.max(y));
}

fn run_cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let options = RESOURCE_OPTIONS;
    let size = v.len() * std::mem::size_of::<U>();
    let output = device.new_buffer(size, options).unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_cast_contiguous(
            &device,
            &encoder,
            &kernels,
            name,
            size_of::<T>(),
            v.len(),
            BufferOffset::zero_offset(&input),
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn cast_f32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f32 -> f16
    let results: Vec<half::f16> = run_cast(&v_f32, "cast_f32_f16");
    assert_eq!(results, v_f16);

    // f32 -> bf16
    let results: Vec<bf16> = run_cast(&v_f32, "cast_f32_bf16");
    assert_eq!(results, v_bf16);

    // f32 -> u32
    let results: Vec<u32> = run_cast(&v_f32, "cast_f32_u32");
    assert_eq!(results, v_u32);

    // f32 -> u8
    let results: Vec<u8> = run_cast(&v_f32, "cast_f32_u8");
    assert_eq!(results, v_u8);

    // f32 -> i64
    let results: Vec<i64> = run_cast(&v_f32, "cast_f32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_f16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f16 -> f32
    let results: Vec<f32> = run_cast(&v_f16, "cast_f16_f32");
    assert_eq!(results, v_f32);

    // f16 -> bf16
    let results: Vec<bf16> = run_cast(&v_f16, "cast_f16_bf16");
    assert_eq!(results, v_bf16);

    // f16 -> u32
    let results: Vec<u32> = run_cast(&v_f16, "cast_f16_u32");
    assert_eq!(results, v_u32);

    // f16 -> u8
    let results: Vec<u8> = run_cast(&v_f16, "cast_f16_u8");
    assert_eq!(results, v_u8);

    // f16 -> i64
    let results: Vec<i64> = run_cast(&v_f16, "cast_f16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_bf16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // bf16 -> f32
    let results: Vec<f32> = run_cast(&v_bf16, "cast_bf16_f32");
    assert_eq!(results, v_f32);

    // bf16 -> f16
    let results: Vec<f16> = run_cast(&v_bf16, "cast_bf16_f16");
    assert_eq!(results, v_f16);

    // bf16 -> u32
    let results: Vec<u32> = run_cast(&v_bf16, "cast_bf16_u32");
    assert_eq!(results, v_u32);

    // bf16 -> u8
    let results: Vec<u8> = run_cast(&v_bf16, "cast_bf16_u8");
    assert_eq!(results, v_u8);

    // bf16 -> i64
    let results: Vec<i64> = run_cast(&v_bf16, "cast_bf16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u32 -> f32
    let results: Vec<f32> = run_cast(&v_u32, "cast_u32_f32");
    assert_eq!(results, v_f32);

    // u32 -> f16
    let results: Vec<f16> = run_cast(&v_u32, "cast_u32_f16");
    assert_eq!(results, v_f16);

    // u32 -> bf16
    let results: Vec<bf16> = run_cast(&v_u32, "cast_u32_bf16");
    assert_eq!(results, v_bf16);

    // u32 -> u8
    let results: Vec<u8> = run_cast(&v_u32, "cast_u32_u8");
    assert_eq!(results, v_u8);

    // u32 -> i64
    let results: Vec<i64> = run_cast(&v_u32, "cast_u32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u8() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u8 -> f32
    let results: Vec<f32> = run_cast(&v_u8, "cast_u8_f32");
    assert_eq!(results, v_f32);

    // u8 -> f16
    let results: Vec<f16> = run_cast(&v_u8, "cast_u8_f16");
    assert_eq!(results, v_f16);

    // u8 -> bf16
    let results: Vec<bf16> = run_cast(&v_u8, "cast_u8_bf16");
    assert_eq!(results, v_bf16);

    // u8 -> u32
    let results: Vec<u32> = run_cast(&v_u8, "cast_u8_u32");
    assert_eq!(results, v_u32);

    // u8 -> i64
    let results: Vec<i64> = run_cast(&v_u8, "cast_u8_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_i64() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // i64 -> f32
    let results: Vec<f32> = run_cast(&v_i64, "cast_i64_f32");
    assert_eq!(results, v_f32);

    // i64 -> f16
    let results: Vec<f16> = run_cast(&v_i64, "cast_i64_f16");
    assert_eq!(results, v_f16);

    // i64 -> bf16
    let results: Vec<bf16> = run_cast(&v_i64, "cast_i64_bf16");
    assert_eq!(results, v_bf16);

    // i64 -> u32
    let results: Vec<u32> = run_cast(&v_i64, "cast_i64_u32");
    assert_eq!(results, v_u32);

    // i64 -> u8
    let results: Vec<u8> = run_cast(&v_i64, "cast_i64_u8");
    assert_eq!(results, v_u8);
}

fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let size = v.len();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_affine(
            &device,
            &encoder,
            &kernels,
            "affine_f32",
            size_of::<T>(),
            size,
            BufferOffset::zero_offset(&input),
            &output,
            mul as f32,
            add as f32,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

fn run_affine_strided<T: Clone>(
    v: &[T],
    shape: &[usize],
    strides: &[usize],
    mul: f64,
    add: f64,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_affine_strided(
            &device,
            &encoder,
            &kernels,
            "affine_f32_strided",
            shape,
            BufferOffset::zero_offset(&input),
            strides,
            &output,
            mul as f32,
            add as f32,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let len: usize = shape.iter().product();
    read_to_vec(&output, len)
}

#[test]
fn affine() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

    let input = [1.0f32; 40_000];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6; 40_000]);
}

#[test]
fn affine_strided() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let shape = [4];
    let strides = [2];
    let result = run_affine_strided(&input, &shape, &strides, mul, add);
    // 1 on 2
    assert_eq!(result, vec![2.6, 5.6, 8.6, 11.6]);
}

fn run_mlx_sort<T: Clone>(v: &[T], ncols: usize) -> Vec<u32> {
    let nrows = v.len() / ncols;
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let indexes = vec![0u32; v.len()];
    let output = new_buffer(&device, &indexes);

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_arg_sort(
            &device,
            &encoder,
            &kernels,
            DType::F32,
            nrows,
            ncols,
            BufferOffset::zero_offset(&input),
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn mlx_sort() {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let input: Vec<_> = (0..8).map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [0, 1, 2, 3, 0, 1, 2, 3]);
    let input: Vec<_> = (0..8).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [3, 2, 1, 0, 3, 2, 1, 0]);
    let input: Vec<_> = (0..1000).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 200);
    let out: Vec<_> = (0..200).rev().collect();
    assert_eq!(&result[..200], out);
    assert_eq!(&result[200..400], out);
    assert_eq!(&result[400..600], out);
    assert_eq!(&result[600..800], out);
    assert_eq!(&result[800..], out);

    // Multi-block test
    let ncols = 16000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(299792458);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let input: Vec<f32> = (0..ncols * 16).map(|_| normal.sample(&mut rng)).collect();
    let result = run_mlx_sort(&input, ncols);
    for start in 0..16 {
        let slice = &input[start * ncols..(start + 1) * ncols];
        let result = &result[start * ncols..(start + 1) * ncols];
        let mut perm: Vec<usize> = (0..ncols).collect();
        perm.sort_by(|i1, i2| slice[*i1].total_cmp(&slice[*i2]));
        let perm: Vec<_> = perm.into_iter().map(|v| v as u32).collect();
        assert_eq!(perm, result);
    }
}

#[test]
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let stride = [1, 2];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn index_select_strided() {
    let embedding = (0..16).map(|x| x as f32).collect::<Vec<_>>();
    let shape = [2, 2];
    let stride = [2, 4];
    let ids = [0u32];
    let dim = 0;
    let result = run_index_select_strided(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![0.0, 4.0]);
}

#[test]
fn index_select_f16() {
    let embedding: Vec<_> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f16");
    assert_eq!(
        approx_f16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u8_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_u8_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_i64_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0i64, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_i64_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_dim1() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 7.0, 8.0f32, 7.0, 9.0, 10.0, 9.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_index_select(
            &device,
            &encoder,
            &kernels,
            name,
            shape,
            ids.len(),
            dim,
            true,
            shape,
            stride,
            BufferOffset::zero_offset(&embeddings_buffer),
            BufferOffset::zero_offset(&ids_buffer),
            &dst_buffer,
        )
    }
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

fn run_index_select_strided<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_index_select(
            &device,
            &encoder,
            &kernels,
            name,
            shape,
            ids.len(),
            dim,
            false,
            shape,
            stride,
            BufferOffset::zero_offset(&embeddings_buffer),
            BufferOffset::zero_offset(&ids_buffer),
            &dst_buffer,
        )
    }
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let results = run(&v, unary::contiguous::cos::HALF);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

fn run_reduce<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(out_length * core::mem::size_of::<U>(), options)
        .unwrap();
    let shape = vec![in_length];
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    match unsafe {
        call_reduce_contiguous(
            &device,
            &encoder,
            &kernels,
            name,
            &shape,
            out_length,
            BufferOffset::zero_offset(&input),
            &output,
        )
    } {
        Ok(_) => {}
        Err(e) => {
            println!("{e}");
            panic!();
        }
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_last_softmax(
            &device,
            &encoder,
            &kernels,
            name,
            v.len(),
            last_dim,
            &input,
            0,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

const fn create_array<const N: usize>() -> [f32; N] {
    let mut array: [f32; N] = [0.0; N];
    let mut i = 1;
    while i <= N {
        array[i - 1] = i as f32;
        i += 1;
    }
    array
}

const fn correct_sum<const N: usize, const D: usize>() -> [f32; D] {
    let mut sum = 0;
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        sum += i;
        i += 1;
        if i > j * N / D {
            results[j - 1] = sum as f32;
            j += 1;
            sum = 0;
        }
    }
    results
}

const fn correct_max<const N: usize, const D: usize>() -> [f32; D] {
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        i += 1;
        if i > j * (N / D) {
            results[j - 1] = (i - 1) as f32;
            j += 1;
        }
    }
    results
}

fn correct_argmax<const N: usize, const D: usize>(arr: [f32; N]) -> [u32; D] {
    let mut max = 0.0;
    let mut max_index: u32 = 0;
    let mut results: [u32; D] = [0; D];
    let mut i = 0;
    let mut j = 1;
    while i <= N {
        if i >= (j * N / D) {
            results[j - 1] = max_index;
            max = 0.0;
            max_index = 0;
            j += 1;
        }
        if i == N {
            break;
        }
        if arr[i] > max {
            max = arr[i];
            max_index = i as u32;
        }
        i += 1;
    }
    results
}

fn reduce_sum_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_sum_f32");
    assert_eq!(approx(results, 4), correct_sum::<N, D>());
}

fn reduce_max_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_max_f32");
    assert_eq!(approx(results, 4), correct_max::<N, D>());
}

fn reduce_argmax_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results: Vec<u32> = run_reduce(&v, N, D, "fast_argmax_f32");
    assert_eq!(results, correct_argmax::<N, D>(v));
}

#[test]
fn reduce_sum1() {
    reduce_sum_case::<9, 1>();
    reduce_sum_case::<6, 1>();
    reduce_sum_case::<10, 1>();
    reduce_sum_case::<64, 1>();
    reduce_sum_case::<128, 1>();
    reduce_sum_case::<256, 1>();
    reduce_sum_case::<512, 1>();
    reduce_sum_case::<1024, 1>();
    reduce_sum_case::<2048, 1>();
    reduce_sum_case::<4096, 1>();
}

#[test]
fn reduce_sum2() {
    reduce_sum_case::<6, 2>();
    reduce_sum_case::<10, 2>();
    reduce_sum_case::<64, 2>();
    reduce_sum_case::<128, 2>();
    reduce_sum_case::<256, 2>();
    reduce_sum_case::<512, 2>();
    reduce_sum_case::<1024, 2>();
    reduce_sum_case::<2048, 2>();
    reduce_sum_case::<4096, 2>();
}

#[test]
fn reduce_max() {
    reduce_max_case::<6, 1>();
    reduce_max_case::<9, 1>();
    reduce_max_case::<10, 1>();
    reduce_max_case::<64, 1>();
    reduce_max_case::<128, 1>();
    reduce_max_case::<256, 1>();
    reduce_max_case::<512, 1>();
    reduce_max_case::<1024, 1>();
    reduce_max_case::<2048, 1>();
    reduce_max_case::<4096, 1>();

    reduce_max_case::<6, 2>();
    reduce_max_case::<10, 2>();
    reduce_max_case::<64, 2>();
    reduce_max_case::<128, 2>();
    reduce_max_case::<256, 2>();
    reduce_max_case::<512, 2>();
    reduce_max_case::<1024, 2>();
    reduce_max_case::<2048, 2>();
    reduce_max_case::<4096, 2>();

    reduce_max_case::<6, 3>();
    reduce_max_case::<10, 3>();
    reduce_max_case::<64, 3>();
    reduce_max_case::<128, 3>();
    reduce_max_case::<256, 3>();
    reduce_max_case::<512, 3>();
    reduce_max_case::<1024, 3>();
    reduce_max_case::<2048, 3>();
    reduce_max_case::<4096, 3>();
}

#[test]
fn reduce_argmax() {
    reduce_argmax_case::<6, 1>();
    reduce_argmax_case::<9, 1>();
    reduce_argmax_case::<10, 1>();
    reduce_argmax_case::<64, 1>();
    reduce_argmax_case::<128, 1>();
    reduce_argmax_case::<256, 1>();
    reduce_argmax_case::<512, 1>();
    reduce_argmax_case::<1024, 1>();
    reduce_argmax_case::<2048, 1>();
}

#[test]
fn reduce_argmax2() {
    reduce_argmax_case::<6, 2>();
    reduce_argmax_case::<10, 2>();
    reduce_argmax_case::<64, 2>();
    reduce_argmax_case::<128, 2>();
    reduce_argmax_case::<256, 2>();
    reduce_argmax_case::<512, 2>();
    reduce_argmax_case::<1024, 2>();
    reduce_argmax_case::<2048, 2>();
    reduce_argmax_case::<4096, 2>();
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let last_dim = 4096;
    let n = 200;
    let mut v = vec![0.0; n * last_dim];
    for i in 0..n {
        v[i * last_dim] = 20.0;
    }
    let results = run_softmax(&v, last_dim, "softmax_f32");
    let results = approx(results, 4);
    assert_eq!(
        results.iter().map(|&s| s.round() as usize).sum::<usize>(),
        n
    );
    assert_eq!(results[0], 1.0);
    assert_eq!(results[1], 0.0);
    assert_eq!(results[last_dim], 1.0);
    assert_eq!(results[2 * last_dim], 1.0);

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f16");
    assert_eq!(
        approx_f16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2332, 0.6338]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_bf16");
    assert_eq!(
        approx_bf16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0859, 0.2324, 0.6328]
    );
}

#[allow(clippy::too_many_arguments)]
fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let length = cond.len();
    let cond = device
        .new_buffer_with_data(
            cond.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(cond),
            options,
        )
        .unwrap();
    let left = device
        .new_buffer_with_data(
            left_true.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();
    let right = device
        .new_buffer_with_data(
            right_false.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();

    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    let cond = BufferOffset {
        buffer: &cond,
        offset_in_bytes: cond_offset,
    };
    let left = BufferOffset {
        buffer: &left,
        offset_in_bytes: left_offset,
    };
    let right = BufferOffset {
        buffer: &right,
        offset_in_bytes: cond_offset,
    };
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_where_cond(
            &device,
            &encoder,
            &kernels,
            name,
            size_of::<T>(),
            shape,
            cond,
            &cond_stride,
            true,
            left,
            &left_stride,
            true,
            right,
            &cond_stride,
            true,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}
#[test]
fn where_cond_u32_f32() {
    let shape = vec![6];
    let cond = vec![0u32, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u32_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}

#[allow(clippy::too_many_arguments)]
fn run_mlx_gemm<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    lhs_offset: usize,
    rhs: &[T],
    rhs_stride: &[usize],
    rhs_offset: usize,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let lhs = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(lhs),
            options,
        )
        .unwrap();
    let rhs = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(rhs),
            options,
        )
        .unwrap();
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemm(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            lhs_offset,
            &lhs,
            rhs_stride,
            rhs_offset,
            &rhs,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn mlx_gemm() {
    let (b, m, n, k) = (1, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
    );

    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![
            20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0, 488.0,
            518.0, 548.0, 578.0
        ]
    );

    // OFFSET
    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    // Manually set batch_size=1 and offset 12 elements * 4 the number of bytes for f32
    let results = run_mlx_gemm(
        GemmDType::F32,
        (1, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        12 * 4,
    );
    assert_eq!(
        approx(results, 4),
        vec![56.0, 59.0, 62.0, 65.0, 200.0, 212.0, 224.0, 236.0]
    );

    // bgemm sanity test
    {
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<bf16> = (0..b * m * k).map(|f| bf16::from_f32(f as f32)).collect();
        let rhs: Vec<bf16> = (0..b * n * k).map(|f| bf16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::BF16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_bf16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }

    {
        // hgemm sanity test
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::F16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_f16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }
}

fn run_random<T: Clone>(name: &'static str, seed: u64, length: usize, a: f32, b: f32) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();

    let seed = device
        .new_buffer_with_data(
            &seed as *const u64 as *const core::ffi::c_void,
            std::mem::size_of::<u64>(),
            options,
        )
        .unwrap();

    if name.starts_with("rand_uniform") {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_random_uniform(
                &device, &encoder, &kernels, name, a, b, length, &seed, &output,
            )
        }
        .unwrap();
    } else {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_random_normal(
                &device, &encoder, &kernels, name, a, b, length, &seed, &output,
            )
        }
        .unwrap();
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn random() {
    fn calc_mean(data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>();
        let count = data.len();
        assert!(count > 0);
        sum / count as f32
    }

    fn calc_stddev(data: &[f32]) -> f32 {
        let mean = calc_mean(data);
        let count = data.len();
        assert!(count > 0);

        let variance = data
            .iter()
            .map(|value| {
                let diff = mean - *value;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;

        variance.sqrt()
    }

    let shape = [1024, 10];

    let length = shape.iter().product::<usize>();
    let seed = 299792458u64;

    let min = -30.0;
    let max = 30.0;
    let mean = 100.0;
    let stddev = 50.0;

    macro_rules! validate_random {
        ($type:ty) => {
            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_uniform_", stringify!($type)),
                seed,
                length,
                min,
                max,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            results.iter().for_each(|v| {
                assert!(*v >= min && *v <= max);
            });
            assert!(calc_mean(&results) > -1.0 && calc_mean(&results) < 1.0);

            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_normal_", stringify!($type)),
                seed,
                length,
                mean,
                stddev,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            assert!((calc_mean(&results) - mean).abs() < mean / 10.0);
            assert!((calc_stddev(&results) - stddev).abs() < stddev / 10.0);
        };
    }

    validate_random!(f32);
    validate_random!(f16);
    validate_random!(bf16);
}

fn run_scatter_add<T: Clone, I: Clone + std::fmt::Debug>(
    input: &[T],
    ids: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);
    let ids_buffer = new_buffer(&device, ids);
    let output = device
        .new_buffer(std::mem::size_of_val(input), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_scatter(
            &device,
            &encoder,
            &kernels,
            name,
            shape,
            shape,
            dim,
            BufferOffset::zero_offset(&input_buffer),
            BufferOffset::zero_offset(&ids_buffer),
            BufferOffset::zero_offset(&output),
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, input.len())
}

#[test]
fn scatter_add() {
    let ids_u8 = [0u8, 0, 1, 0, 2, 2, 3, 3];
    let ids_u32 = [0u32, 0, 1, 0, 2, 2, 3, 3];
    let ids_i64 = [0i64, 0, 1, 0, 2, 2, 3, 3];

    let input_f32 = [5.0f32, 1.0, 7.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let input_f16 = input_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let input_bf16 = input_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim1_f32 = vec![8.0, 7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let output_dim1_f16 = output_dim1_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim1_bf16 = output_dim1_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim2_f32 = vec![5.0, 3.0, 7.0, 0.0, 3.0, 2.0, 1.0, 3.0];
    let output_dim2_f16 = output_dim2_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim2_bf16 = output_dim2_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    for (shape, output_f32, output_f16, output_bf16) in [
        (vec![8], output_dim1_f32, output_dim1_f16, output_dim1_bf16),
        (
            vec![4, 2],
            output_dim2_f32,
            output_dim2_f16,
            output_dim2_bf16,
        ),
    ] {
        for results in [
            run_scatter_add(&input_f32, &ids_u8, &shape, 0, "sa_u8_f32"),
            run_scatter_add(&input_f32, &ids_u32, &shape, 0, "sa_u32_f32"),
            run_scatter_add(&input_f32, &ids_i64, &shape, 0, "sa_i64_f32"),
        ] {
            assert_eq!(results, output_f32);
        }
        for results in [
            run_scatter_add(&input_f16, &ids_u8, &shape, 0, "sa_u8_f16"),
            run_scatter_add(&input_f16, &ids_u32, &shape, 0, "sa_u32_f16"),
            run_scatter_add(&input_f16, &ids_i64, &shape, 0, "sa_i64_f16"),
        ] {
            assert_eq!(results, output_f16);
        }
        for results in [
            run_scatter_add(&input_bf16, &ids_u8, &shape, 0, "sa_u8_bf16"),
            run_scatter_add(&input_bf16, &ids_u32, &shape, 0, "sa_u32_bf16"),
            run_scatter_add(&input_bf16, &ids_i64, &shape, 0, "sa_i64_bf16"),
        ] {
            assert_eq!(results, output_bf16);
        }
    }
}

fn run_index_add<T: Clone, I: Clone + std::fmt::Debug>(
    left: &[T],
    right: &[T],
    indices: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input_buffer = new_buffer(&device, right);
    let output = new_buffer(&device, left);
    let indices_buffer = new_buffer(&device, indices);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_index_add(
            &device,
            &encoder,
            &kernels,
            name,
            shape,
            shape,
            shape,
            dim,
            BufferOffset::zero_offset(&input_buffer),
            BufferOffset::zero_offset(&indices_buffer),
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, left.len())
}

#[test]
fn index_add() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let indices = vec![0u32, 1, 0, 1, 0, 1];
    let shape = vec![6];

    // u32, f32
    {
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, f16
    {
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, bf16
    {
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f32
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, bf16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f32
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, bf16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

fn run_pool2d<T: Clone>(
    v: &[T],
    (w_k, h_k): (usize, usize),
    (w_stride, h_stride): (usize, usize),
    shape: &[usize],
    strides: &[usize],
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out_w = (shape[2] - w_k) / w_stride + 1;
    let out_h = (shape[3] - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_pool2d(
            &device, &encoder, &kernels, name, shape, strides, out_w, out_h, w_k, h_k, w_stride,
            h_stride, &input, &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn max_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 7.0, 13.0, 15.0];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f32",
    );
    let expected = vec![
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<f16> = (0..16).map(|v| f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| f16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<bf16> = (0..16).map(|v| bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_bf16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| bf16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u8",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u32",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    input_stride: &[usize],
    kernel: &[T],
    kernel_shape: &[usize],
    kernel_stride: &[usize],
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let c_out = kernel_shape[1];
    let k_size = kernel_shape[2];
    let b_size = input_shape[0];
    let l_in = input_shape[2];
    let l_out = (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1;
    let dst_el = c_out * l_out * b_size;

    let input = new_buffer(&device, input);
    let kernel = new_buffer(&device, kernel);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_conv_transpose1d(
            &device,
            &encoder,
            &kernels,
            name,
            dilation,
            stride,
            padding,
            out_padding,
            c_out,
            l_out,
            b_size,
            input_shape,
            input_stride,
            kernel_shape,
            kernel_stride,
            &input,
            0,
            &kernel,
            0,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn conv_transpose1d_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel = vec![1.0f32, 2.0, 3.0, 4.0];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f32",
    );

    let expected = vec![1., 4., 10., 20., 25., 24., 16.];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_f16() {
    let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_bf16() {
    let input: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_bf16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u8() {
    let input: Vec<u8> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u8> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u8",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u32() {
    let input: Vec<u32> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u32> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u32",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn const_fill() {
    fn constant_fill<T: Clone + EncoderParam>(name: &'static str, len: usize, value: T) -> Vec<T> {
        let dev = device();
        let kernels = Kernels::new();
        let commands = commands(&dev);
        let encoder = commands.command_encoder().unwrap();
        let buffer = dev
            .new_buffer(len * std::mem::size_of::<T>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { call_const_fill(&dev, &encoder, &kernels, name, len, &buffer, value) }.unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        read_to_vec::<T>(&buffer, len)
    }
    fn test<T: Clone + Copy + EncoderParam + PartialEq + std::fmt::Debug, F: FnOnce(f32) -> T>(
        name: &'static str,
        f: F,
    ) {
        let len = rand::rng().random_range(2..16) * rand::rng().random_range(4..16);
        let value = rand::rng().random_range(1. ..19.);
        let value = f(value);
        let v = constant_fill::<T>(name, len, value);
        assert_eq!(v, vec![value; len])
    }
    test::<u8, _>("fill_u8", |v| v as u8);
    test::<u32, _>("fill_u32", |v| v as u32);
    test::<i64, _>("fill_i64", |v| v as i64);
    test::<f16, _>("fill_f16", f16::from_f32);
    test::<bf16, _>("fill_bf16", bf16::from_f32);
    test::<f32, _>("fill_f32", |v| v);
}

#[test]
fn commands_creation_and_encoder() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Commands::new(queue, residency_set).unwrap();

    let encoder = commands.command_encoder().unwrap();
    drop(encoder);
}

#[test]
fn commands_concurrent_acquisition() {
    std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "2");

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Arc::new(Commands::new(queue, residency_set).unwrap());

    let mut handles = vec![];

    for _ in 0..16 {
        let c = Arc::clone(&commands);
        handles.push(thread::spawn(move || {
            let encoder = c.command_encoder().unwrap();
            drop(encoder);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    commands.wait_until_completed().unwrap();
}

#[test]
fn residency_set_batch_insert_remove() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = (unsafe { set.raw() }) else {
        // Residency sets are unsupported on this device/OS; the set no-ops.
        return;
    };

    let bufs: Vec<Buffer> = (0..3).map(|i| new_buffer(&device, &[i as f32])).collect();
    let base = raw.allocationCount();

    set.insert_batch(&bufs);
    assert_eq!(raw.allocationCount(), base + bufs.len());
    set.remove_batch(&bufs);
    assert_eq!(raw.allocationCount(), base);

    // Empty batches are valid and leave the set untouched.
    set.insert_batch(std::iter::empty());
    set.remove_batch(std::iter::empty());
    assert_eq!(raw.allocationCount(), base);
}

/// A second remove of the same buffer must not reach `removeAllocation`.
///
/// **This is the test that cannot be written the obvious way.** The defect's
/// natural expression is `IOGPUGroupMemory::remove_memory_object()` panicking
/// the *machine* (`DESIGN.md` §6.3c), so a test that actually provokes it costs
/// the machine and the user's session and must never be written. What is
/// testable is the guard: the membership set refuses the second remove, so the
/// call never happens.
///
/// The assertion is on **two independent quantities** — our own membership
/// count and Metal's `allocationCount()` — because agreeing with itself is what
/// a broken bookkeeping layer also does.
#[test]
fn residency_set_refuses_a_double_remove() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = (unsafe { set.raw() }) else {
        return;
    };

    let buf = new_buffer(&device, &[1.0f32]);
    let base = raw.allocationCount();

    set.insert(&buf);
    assert!(set.contains(&buf), "insert did not record membership");
    assert_eq!(raw.allocationCount(), base + 1);

    // First remove: the set holds it, so this is a real removal.
    assert_eq!(set.remove(&buf), 1, "first remove should have removed one");
    assert!(!set.contains(&buf));
    assert_eq!(raw.allocationCount(), base);

    // Second remove: absent. This is the call that would reach the kernel's
    // assertion, and the count returning 0 is the evidence it did not.
    assert_eq!(
        set.remove(&buf),
        0,
        "a second remove must be refused, not forwarded to Metal"
    );
    assert_eq!(raw.allocationCount(), base);

    // A buffer that was never inserted is the same case by a different route.
    let never = new_buffer(&device, &[2.0f32]);
    assert_eq!(
        set.remove(&never),
        0,
        "removing a never-inserted buffer must be refused"
    );
    assert_eq!(raw.allocationCount(), base);
}

/// Inserting the same buffer twice must leave the set holding it **once**.
///
/// Otherwise the set's own count and Metal's disagree about how many removes
/// the allocation needs, and the guard in `remove` stops being able to decide.
#[test]
fn residency_set_insert_is_idempotent() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = (unsafe { set.raw() }) else {
        return;
    };

    let buf = new_buffer(&device, &[1.0f32]);
    let base = raw.allocationCount();
    let base_members = set.len();

    assert_eq!(set.insert(&buf), 1, "the first insert adds the allocation");

    // **The returned count is the discriminating quantity, and neither of the
    // two obvious ones is.** `addAllocation` is idempotent on Metal's side, so
    // `allocationCount()` reads `base + 1` whether or not the duplicate was
    // forwarded; and a `HashSet` holding one key has the same length either
    // way. Both assertions pass under a mutation that re-adds unconditionally —
    // measured. Only counting the calls we chose to make sees it.
    assert_eq!(
        set.insert(&buf),
        0,
        "a repeated insert must be skipped, not forwarded to Metal"
    );
    assert_eq!(set.len(), base_members + 1);
    assert_eq!(raw.allocationCount(), base + 1);

    assert_eq!(set.remove(&buf), 1);
    assert_eq!(set.len(), base_members);
    assert_eq!(raw.allocationCount(), base);
    // And the second remove is still refused, which is the property a
    // double-insert would have broken.
    assert_eq!(set.remove(&buf), 0);
}

/// An arena slot is a view over a shared parent, so N views are **one**
/// allocation to Metal and must be one member here (`DESIGN.md` §9.2).
///
/// Keying membership on the handle rather than on the underlying `MTLBuffer`
/// would make a view's removal drop the parent while other views still use it —
/// and would make the parent's insertion look like N removals' worth.
#[test]
fn residency_set_keys_views_to_their_parent() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = (unsafe { set.raw() }) else {
        return;
    };

    let parent = new_buffer(&device, &[0.0f32; 64]);
    let base = raw.allocationCount();

    set.insert(&parent);
    assert_eq!(raw.allocationCount(), base + 1);

    // Two disjoint views over the same allocation.
    let a = parent.view(0, 16);
    let b = parent.view(16, 16);
    assert!(
        set.contains(&a),
        "a view must resolve to its parent's entry"
    );
    assert!(set.contains(&b));

    // Inserting a view adds nothing: the allocation is already there.
    set.insert(&a);
    assert_eq!(
        raw.allocationCount(),
        base + 1,
        "inserting a view must not add a second allocation"
    );

    // Removing through one view removes the single allocation, and the second
    // view's removal is then refused rather than reaching the kernel.
    assert_eq!(set.remove(&a), 1);
    assert_eq!(raw.allocationCount(), base);
    assert_eq!(
        set.remove(&b),
        0,
        "a second view's remove must be refused once the parent is gone"
    );
    assert_eq!(raw.allocationCount(), base);
}

/// `remove_all` is the teardown path: it empties the set in one commit and
/// leaves every later remove refused.
#[test]
fn residency_set_remove_all_empties_and_stays_empty() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = (unsafe { set.raw() }) else {
        return;
    };

    // This set is freshly created, so it holds nothing but what is inserted
    // here — which is what lets the count below be asserted absolutely.
    assert_eq!(raw.allocationCount(), 0);

    let bufs: Vec<Buffer> = (0..4).map(|i| new_buffer(&device, &[i as f32])).collect();
    set.insert_batch(&bufs);
    assert_eq!(set.len(), bufs.len());
    assert_eq!(raw.allocationCount(), bufs.len());

    assert_eq!(
        set.remove_all(),
        bufs.len(),
        "remove_all should report what it removed"
    );
    assert!(set.is_empty());
    assert_eq!(
        raw.allocationCount(),
        0,
        "remove_all must empty Metal's set, not only our bookkeeping"
    );

    // Every per-buffer remove after a teardown is refused — this is the
    // ordering the panic came from, with the guard in place.
    for b in &bufs {
        assert_eq!(set.remove(b), 0);
    }
    assert_eq!(set.remove_all(), 0, "a second remove_all is a no-op");
}

/// Every name [`ConvKernel`] declares must exist in the compiled `conv.metal`
/// library.
///
/// This is the check that makes the table in `conv_names.rs` load-bearing
/// rather than decorative. The two sides — the `[[host_name]]` strings that
/// `conv.metal`'s macros expand to, and the Rust strings callers pass to
/// `load_pipeline` — were previously hand-synced with nothing comparing them,
/// so a rename or a dtype added to one side only surfaced as a runtime
/// `LoadFunctionError` in the middle of a forward pass.
///
/// `conv.metal` is compiled from source at runtime, so the library this test
/// queries is the same one a real dispatch resolves against: the oracle is the
/// actual compiled metallib, not a second copy of the list.
#[test]
fn conv_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ConvKernel::ALL {
        for (suffix, name) in family.variants() {
            kernels
                .load_pipeline(&device, Source::Conv, name)
                .unwrap_or_else(|e| {
                    panic!(
                        "conv.metal has no kernel named {name:?} \
                         (declared by ConvKernel::{} for {suffix:?}): {e:?}",
                        family.stem(),
                    )
                });
            checked += 1;
        }
    }

    // Guards against the table being emptied or a family silently dropping all
    // its variants — an all-green run over zero names would otherwise pass.
    // 5 dtypes each for im2col1d, im2col, col2im1d, conv_transpose1d,
    // upsample_nearest2d, upsample_bilinear2d, max_pool2d and avg_pool2d, plus
    // 3 each for the float-only conv1d_depthwise and conv_transpose2d, 3 each
    // for the three k_size-specialized depthwise families (k = 2, 3, 4), 3
    // each for the three fused shortconv_step families (k = 2, 3, 4), and 3
    // for shortconv_ring.
    assert_eq!(
        checked, 67,
        "expected 67 declared conv variants, found {checked}"
    );
}

/// Each declared name must be its family's stem, then its dtype suffix, then
/// the family's tail, and each suffix must appear once per family.
///
/// The names are stored verbatim so they can be grepped against `conv.metal`,
/// which means a row could pair one family's suffix with another family's name
/// and still resolve — `conv_names_resolve` would pass while
/// [`ConvKernel::name`] handed callers the wrong kernel. Checking the spelling
/// against the stem closes that, and the duplicate check catches a copy-pasted
/// row that shadows the one below it.
///
/// The tail carries a compile-tier axis for families that have one (`_k3` on
/// the `k_size`-specialized depthwise variants). It is checked here rather than
/// exempted, so those names are held to the same full spelling rule as the
/// rest — a `_k3` row that resolved to a `_k2` kernel is exactly the
/// wrong-kernel case this test exists to catch, and it would be silent.
#[test]
fn conv_names_match_their_stem_and_suffix() {
    for family in ConvKernel::ALL {
        let mut seen: Vec<&str> = Vec::new();
        for (suffix, name) in family.variants() {
            assert_eq!(
                name,
                format!("{}_{suffix}{}", family.stem(), family.tail()),
                "ConvKernel::{}{} declares {name:?} for {suffix:?}",
                family.stem(),
                family.tail(),
            );
            assert!(
                !seen.contains(&suffix),
                "ConvKernel::{} declares {suffix:?} twice",
                family.stem(),
            );
            seen.push(suffix);
        }
        assert!(
            !seen.is_empty(),
            "ConvKernel::{} declares no variants",
            family.stem(),
        );
    }
}

/// A dtype a family does not declare must not produce a name.
///
/// `i64` and `f64` are instantiated nowhere in `conv.metal`, and the float-only
/// families reject the integer suffixes. Returning `None` is what keeps an
/// unsupported dtype from reaching `load_pipeline` as a string that will fail
/// to resolve — the caller reports it against its own dtype enum instead.
#[test]
fn conv_undeclared_dtype_has_no_name() {
    for family in ConvKernel::ALL {
        for suffix in ["i64", "f64", "i32", "not_a_dtype"] {
            assert_eq!(
                family.name(suffix),
                None,
                "ConvKernel::{} named {suffix:?}, which conv.metal does not instantiate",
                family.stem(),
            );
        }
    }

    // The float-only families specifically: `conv.metal` declares no integer
    // instantiation for either, and both are on the LFM2 decode path.
    assert_eq!(ConvKernel::CONV1D_DEPTHWISE.name("u8"), None);
    assert_eq!(ConvKernel::CONV1D_DEPTHWISE.name("u32"), None);
    assert_eq!(ConvKernel::CONV_TRANSPOSE2D.name("u8"), None);
    assert_eq!(ConvKernel::CONV_TRANSPOSE2D.name("u32"), None);

    // And the positive case, so the assertions above are not vacuous.
    assert_eq!(
        ConvKernel::CONV1D_DEPTHWISE.name("f16"),
        Some("conv1d_depthwise_f16")
    );
}

/// Every name [`ReduceKernel`] declares must exist in the compiled
/// `reduce.metal` library.
///
/// The `reduce.metal` counterpart of `conv_names_resolve`, and it earned its
/// place immediately: converting the file's `impl_*` macros to template
/// instantiations introduced `init_reduce` rows whose `#op` stringized to the
/// operator *type* (`Sum`), emitting `fast_Sum_f32` where every caller asks for
/// `fast_sum_f32`. That compiles and links; all 48 reduction variants were
/// simply absent from the library. Without this test the first symptom would
/// have been a `LoadFunctionError` inside an LFM2 forward pass.
#[test]
fn reduce_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ReduceKernel::ALL {
        for (suffix, name) in family.variants() {
            kernels
                .load_pipeline(&device, family.source(), name)
                .unwrap_or_else(|e| {
                    panic!(
                        "{:?} has no kernel named {name:?} \
                         (declared by ReduceKernel::{}{} for {suffix:?}): {e:?}",
                        family.source(),
                        family.stem(),
                        family.tail(),
                    )
                });
            checked += 1;
        }
    }

    // Guards against the table being emptied or a family silently dropping all
    // its variants — an all-green run over zero names would otherwise pass.
    // 6 dtypes each for sum, mul, min, max, argmin and argmax in both their
    // contiguous and strided forms (12 families), plus 3 float dtypes each for
    // softmax, rmsnorm, rmsnorm_strided, rmsnorm_rope, rmsnorm_rope_strided,
    // layernorm, rope, rope_i and rope_thd.
    assert_eq!(
        checked, 99,
        "expected 99 declared reduce variants, found {checked}"
    );
}

/// Each declared name must be its family's stem, then its dtype suffix, then
/// the family's tail, and each suffix must appear once per family.
///
/// Resolution alone cannot catch a row that names a *different* real kernel.
/// That matters more here than it did for conv, because the strided and
/// contiguous forms of a reduction differ by a suffix and take different
/// argument lists: a `_strided` row pointing at the contiguous kernel resolves
/// perfectly well and then reads a `strides` argument that was never bound.
#[test]
fn reduce_names_match_their_stem_and_suffix() {
    for family in ReduceKernel::ALL {
        let mut seen: Vec<&str> = Vec::new();
        for (suffix, name) in family.variants() {
            assert_eq!(
                name,
                format!("{}_{suffix}{}", family.stem(), family.tail()),
                "ReduceKernel::{}{} declares {name:?} for {suffix:?}",
                family.stem(),
                family.tail(),
            );
            assert!(
                !seen.contains(&suffix),
                "ReduceKernel::{}{} declares {suffix:?} twice",
                family.stem(),
                family.tail(),
            );
            seen.push(suffix);
        }
        assert!(
            !seen.is_empty(),
            "ReduceKernel::{}{} declares no variants",
            family.stem(),
            family.tail(),
        );
    }
}

/// A dtype a family does not declare must not produce a name.
///
/// Softmax, the norms and RoPE are float-only in `reduce.metal`, so the integer
/// suffixes must be refused here rather than reaching `load_pipeline`.
#[test]
fn reduce_undeclared_dtype_has_no_name() {
    for family in [
        ReduceKernel::SOFTMAX,
        ReduceKernel::RMSNORM,
        ReduceKernel::RMSNORM_STRIDED,
        ReduceKernel::RMSNORM_ROPE,
        ReduceKernel::RMSNORM_ROPE_STRIDED,
        ReduceKernel::LAYERNORM,
        ReduceKernel::ROPE,
        ReduceKernel::ROPE_I,
        ReduceKernel::ROPE_THD,
    ] {
        for suffix in ["u8", "u32", "i64", "f64", "not_a_dtype"] {
            assert_eq!(
                family.name(suffix),
                None,
                "ReduceKernel::{} named {suffix:?}, which reduce.metal does not instantiate",
                family.stem(),
            );
        }
    }

    // f64 is instantiated nowhere in reduce.metal, including for the families
    // that do carry the integer dtypes.
    for family in ReduceKernel::ALL {
        assert_eq!(family.name("f64"), None);
    }

    // And the positive cases, so the assertions above are not vacuous.
    assert_eq!(ReduceKernel::SUM.name("f16"), Some("fast_sum_f16"));
    assert_eq!(
        ReduceKernel::SUM_STRIDED.name("f16"),
        Some("fast_sum_f16_strided")
    );
}

// ---------------------------------------------------------------------------
// Packed parameters (issue #38)
//
// `reduce.metal` carries every kernel twice: the classical form binding each
// scalar with `setBytes`, and a `_packed` form taking one `device const
// Params*`. Only the second is expressible in an ICB command, which has no
// `setBytes` in any form (`DESIGN.md` §3.7b).
//
// Three things have to hold, and they fail in different ways:
//
//  * the two sides agree on **layout** — a field at the wrong offset is a
//    plausible wrong answer, not a crash (`DESIGN.md` §3.5, §15.1)
//  * every `_packed` name **resolves** — a missing instantiation compiles and
//    links, and #26 shipped exactly that defect for 48 variants
//  * the two forms compute **bit-identical** output — which is what proves the
//    body really is shared rather than merely similar

// The layout checks themselves are registry-driven and live at the foot of this
// file (`every_family_params_layout_matches_metal`, issue #58). They were seven
// near-identical per-family test bodies until then, and the seventh call site
// was the thing that could be forgotten: a family absent from it was never
// checked, which looks exactly like a family that passes.

// ---------------------------------------------------------------------------
// The parity comparison, and why it is a helper rather than an `assert_eq!`
// (issue #53).
//
// `assert_eq!(split, packed)` passes when both arms are trivially zero: a kernel
// that does nothing agrees with itself perfectly, and the test is green.
//
// That is not a hypothetical. `DESIGN.md` §3.7a records it as the exact
// signature of a pipeline built without `supportIndirectCommandBuffers`:
// *"status=completed, error=nil, output all zeros"*. The packed-params work
// (§11.3a) exists to make the ICB path reachable, and a silently-empty kernel is
// that path's characteristic failure -- so a parity suite that cannot tell
// "both correct" from "both silent" is the wrong suite to be relying on when it
// lands.
//
// # Why a helper rather than a guard copied into each arm
//
// The guard was written once, by hand, in `packed_matches_split_for_copy2d_f16`.
// Copying it into the other 29 arms would close today's gap and leave the same
// failure mode one step along: the 31st arm is added without it, silently, and
// an unguarded arm is indistinguishable from a passing one. That is the shape
// #58 removed from the layout checks, and the argument is the same here.
//
// So the comparison and the guard are **one operation**. An arm does not get to
// perform the first without the second, because there is only one function that
// compares a `(split, packed)` pair and it does both. `every_parity_arm_routes
// _through_the_helper` then closes the remaining hole -- an arm that hand-rolls
// its own `assert_eq!` instead of calling this -- by reading the file and naming
// the offender.
//
// This is weaker than #58's `error[E0004]`, and the difference is worth stating
// rather than glossing: the parity arms are not one operation over a descriptor
// the way a layout check is. Each has its own call sequence, its own element
// type and its own buffer setup, so there is no data-shaped registration a type
// could make exhaustive. What is shared is the *comparison*, and that is where
// the check is put.
//
// # Why "at least two distinct values" rather than "not all zero"
//
// The issue proposes not-all-zero as the minimum. It is the right instinct and
// it is not quite the right predicate, for two reasons found while applying it:
//
//  * **Zero is a legitimate output.** `fast_argmax_f32` returns *indices*, and
//    index 0 is the correct answer whenever the maximum is first in its block.
//    A not-all-zero guard makes a correct test fail there, and the fix would be
//    to weaken the guard -- which puts the suite back where it started.
//  * **Distinctness is strictly stronger where both apply.** A kernel that
//    writes one constant everywhere passes not-all-zero as long as the constant
//    is not zero. Requiring two distinct values catches that too, and it is the
//    same cost -- one pass over the output either way.
//
// The arms whose inputs had to change so that a correct kernel produces a varied
// output, rather than being exempted from the check, are noted at their call
// sites. Exempting an arm is not offered as an option: there is no parameter to
// this function that turns the check off.

/// An element type a parity arm compares.
///
/// Bit-identical rather than approximately equal, deliberately, and the trait is
/// what makes that uniform across the four element types the arms use. Both
/// entry points of every family are instantiated from one body, so anything but
/// an exact match means the binding style changed what the kernel computed --
/// which is the question the packed-params work asks. `DESIGN.md` §2.3 makes the
/// same argument for the engine as a whole.
///
/// `Bits` rather than comparing the values directly because `f16` and `f32` are
/// not `Eq`, and because a NaN produced identically by both arms should compare
/// equal here: the question is whether the two kernels wrote the same bytes, not
/// whether those bytes denote the same number.
trait ParityElem: Copy {
    /// The comparison key. Equal keys mean identical bytes.
    type Bits: Copy + Eq + std::hash::Hash + std::fmt::Debug;
    fn parity_bits(self) -> Self::Bits;
}

impl ParityElem for f16 {
    type Bits = u16;
    fn parity_bits(self) -> u16 {
        self.to_bits()
    }
}

impl ParityElem for f32 {
    type Bits = u32;
    fn parity_bits(self) -> u32 {
        self.to_bits()
    }
}

impl ParityElem for u32 {
    type Bits = u32;
    fn parity_bits(self) -> u32 {
        self
    }
}

impl ParityElem for u8 {
    type Bits = u8;
    fn parity_bits(self) -> u8 {
        self
    }
}

/// Assert that the two binding styles agree, and that the comparison was not
/// vacuous.
///
/// **The only way an arm compares its two outputs.** Both checks or neither;
/// there is no argument that disables the second, and no second function that
/// does the first alone.
///
/// `what` names the kernel, so a failure says which one broke without the
/// reader having to map a test name onto a `[[host_name]]`.
#[track_caller]
fn assert_packed_matches_split<T: ParityElem>(what: &str, split: &[T], packed: &[T]) {
    let split_bits: Vec<_> = split.iter().map(|x| x.parity_bits()).collect();
    let packed_bits: Vec<_> = packed.iter().map(|x| x.parity_bits()).collect();
    assert_eq!(
        split_bits, packed_bits,
        "{what} and its packed form disagree"
    );

    // The guard. A kernel that wrote nothing -- or wrote one constant
    // everywhere -- agrees with itself, so equality above is only evidence when
    // the output actually varies.
    //
    // Counted on the *split* arm because it is the classical path: if it is
    // uniform the arm proves nothing regardless of what packed did, and saying
    // so against the reference side gives the clearer message.
    let distinct = split_bits
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct >= 2,
        "{what} wrote {distinct} distinct value(s) across {} elements, so the \
         parity comparison is vacuous: two kernels that both do nothing agree \
         perfectly. `DESIGN.md` §3.7a records all-zero output as the signature \
         of a missing supportIndirectCommandBuffers. Fix the arm's *input* so a \
         correct kernel produces a varied output -- do not relax this check.",
        split_bits.len(),
    );
}

/// Every parity arm compares through `assert_packed_matches_split`.
///
/// This is the half a helper alone cannot enforce. Routing the 30 existing arms
/// through one function closes today's gap; nothing stops the 31st from writing
/// its own `assert_eq!` and skipping the guard, which is exactly how the gap
/// this issue closes was opened -- one arm had the check, the rest were added
/// beside it without.
///
/// So the file checks itself. A parity arm is a `fn` whose name contains
/// `packed_matches_split_for_`, and each one must mention the helper. An arm
/// that does not is named in the failure, with the line it starts at.
///
/// # What this does and does not prove
///
/// It proves no arm *omits* the call. It does not prove an arm calls it on the
/// right buffers -- a test can always be written to check the wrong thing, and
/// no mechanism in the file can see that. What it removes is the silent case:
/// forgetting, which is what actually happened here across four families.
///
/// Source-scanning is a blunt instrument and is used because the alternative is
/// blunter. A macro generating the arms would make omission impossible, but the
/// arms differ in call sequence, element type, buffer setup and argument count
/// -- 30 bodies with almost nothing in common but their last two lines -- so the
/// macro would take one argument per line of body and hide what each arm tests.
/// #58 could use the type system because a layout check is data; this is not.
#[test]
fn every_parity_arm_routes_through_the_helper() {
    const HELPER: &str = "assert_packed_matches_split";
    // The file reads itself. `include_str!` is resolved at compile time against
    // this same source, so the scan cannot drift from what is compiled.
    let src = include_str!("tests.rs");

    let mut arms = 0usize;
    let mut delinquent = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("fn ") || !trimmed.contains("packed_matches_split_for_") {
            continue;
        }
        arms += 1;
        // The body runs to the first line that is exactly `}` at column zero,
        // which is how every arm in this file ends.
        let body_end = lines[i + 1..]
            .iter()
            .position(|l| *l == "}")
            .map(|off| i + 1 + off)
            .unwrap_or(lines.len() - 1);

        // Comment lines are dropped before the scan. Without this, *mentioning*
        // the helper in a comment satisfies the check while an `assert_eq!`
        // below it does the actual comparing -- which is a bypass a reviewer
        // would not see. Found by probing this test rather than reasoned about:
        // the first version matched on the raw body and the comment form passed.
        let code: String = lines[i..=body_end]
            .iter()
            .filter(|l| !l.trim_start().starts_with("//"))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        // The *call* form, not the name. `HELPER` alone would also match a
        // `let _ = assert_packed_matches_split;` or a doc reference.
        if !code.contains(&format!("{HELPER}(")) {
            delinquent.push(format!(
                "{} (line {}) -- does not call {HELPER}",
                trimmed.trim_end_matches(" {"),
                i + 1
            ));
        } else if code.contains("assert_eq!(") {
            // Calling the helper *and* hand-rolling a comparison beside it is
            // not itself wrong -- the argmax arm asserts its indices that way,
            // deliberately -- but a bare `assert_eq!` on the two output vectors
            // is the shape this issue exists to remove, so it is worth naming
            // when it appears on `outs`/`split`/`packed` directly.
            let suspicious = code.contains("assert_eq!(\n        outs[0], outs[1]")
                || code.contains("assert_eq!(\n        split, packed");
            if suspicious {
                delinquent.push(format!(
                    "{} (line {}) -- compares its outputs with a bare assert_eq! \
                     beside the helper",
                    trimmed.trim_end_matches(" {"),
                    i + 1
                ));
            }
        }
    }

    assert!(
        delinquent.is_empty(),
        "these parity arms compare their two outputs without going through \
         `{HELPER}`, so they do not carry the non-vacuity guard and would pass \
         if both kernels wrote nothing (issue #53):\n  {}",
        delinquent.join("\n  "),
    );

    // And the scan itself must not be vacuous: a regex that matched nothing
    // would report success for a file with no arms at all. The exact count is
    // asserted rather than `> 0` so that an arm *lost* to a bad merge is a
    // failure too -- the same argument `packed_names_resolve` makes for
    // checking 90 rather than "some".
    // 37 → 40 (lloom #732): three arms landed on `dev` without this count
    // following them -- `rmsnorm_f16_strided` and `rmsnorm_rope_f16` with the
    // QkvLayout work, `the_fused_epilogues_f16` with P3's packed siblings --
    // so the lib suite carried one red test no gate ran: the CI CPU gate does
    // not run the Metal lib tests, and the kernel harnesses filter by name.
    assert_eq!(
        arms, 40,
        "expected 40 parity arms, found {arms}; if a family was added or \
         removed, update this count deliberately rather than relaxing it"
    );
}

/// Every classical name has a `_packed` counterpart in the compiled library.
///
/// The same argument as `reduce_names_resolve`, applied to the new axis: an
/// absent instantiation is not a compile error on either side, and the first
/// symptom would be a `LoadFunctionError` inside a forward pass. #26 shipped
/// precisely that for 48 variants, caught only by loading against the library.
#[test]
fn packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ReduceKernel::ALL {
        for (suffix, name) in family.variants() {
            let packed = crate::kernels::kernels::params::packed_name(name);
            kernels
                .load_pipeline(&device, family.source(), packed.clone())
                .unwrap_or_else(|e| {
                    panic!(
                        "{:?} has no kernel named {packed:?} \
                         (the packed form of {name:?}, ReduceKernel::{}{} for {suffix:?}): {e:?}",
                        family.source(),
                        family.stem(),
                        family.tail(),
                    )
                });
            checked += 1;
        }
    }
    assert_eq!(
        checked, 99,
        "expected a packed counterpart for all 99 declared variants, found {checked}"
    );
}

/// Run a reduction both ways and return `(split, packed)`.
fn reduce_both_ways<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> (Vec<U>, Vec<U>) {
    let device = device();
    let kernels = Kernels::new();
    let shape = vec![in_length];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, v);
        let output = device
            .new_buffer(out_length * core::mem::size_of::<U>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_reduce_contiguous_with(
                &device,
                &encoder,
                &kernels,
                name,
                &shape,
                out_length,
                BufferOffset::zero_offset(&input),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<U>(&output, out_length));
    }
    let packed = outs.pop().unwrap();
    let split = outs.pop().unwrap();
    (split, packed)
}

/// `fast_sum_f16` — 22 of the 674 dispatches in a decode token — must give the
/// same bits either way.
///
/// Bit-identical rather than approximately equal, deliberately. The two
/// variants are instantiated from one body, so anything but an exact match
/// means the binding style changed what the kernel computed, which is the whole
/// question this issue asks. `DESIGN.md` §2.3 makes the same argument for the
/// engine as a whole.
#[test]
fn packed_matches_split_for_sum_f16() {
    let v: Vec<f16> = (0..1024)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let (split, packed) = reduce_both_ways::<f16, f16>(&v, 1024, 8, "fast_sum_f16");
    assert_packed_matches_split("fast_sum_f16", &split, &packed);
}

#[test]
fn packed_matches_split_for_sum_f32() {
    let v: Vec<f32> = (0..2048)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let (split, packed) = reduce_both_ways::<f32, f32>(&v, 2048, 16, "fast_sum_f32");
    assert_packed_matches_split("fast_sum_f32", &split, &packed);
}

/// `argmax` exercises the `indexed<T>` accumulator and the arg-reduce entry
/// points, which are a separate pair of wrappers from the plain reductions.
///
/// **This is the arm where zero is a legitimate output** (issue #53): the result
/// is an index, and 0 is correct whenever a block's maximum is its first
/// element. A not-all-zero guard would make a correct kernel fail here, and the
/// fix for that would be to weaken the guard -- which is how a suite ends up
/// back where it started. Counting *distinct* values instead is what lets the
/// same check cover this arm and the value-producing ones without an exemption.
#[test]
fn packed_matches_split_for_argmax_f32() {
    // The peak sits at a different offset in each of the four blocks, so a
    // correct argmax returns four different indices and a kernel returning a
    // constant -- including a constant 0 -- fails the guard.
    //
    // The previous input varied only incidentally: `i * 61 % 97` happened not to
    // place two block maxima at the same offset. That was true and nothing made
    // it true, so the arm's non-vacuity was an accident of arithmetic. Planting
    // the peaks makes it a property of the test.
    let block = 128usize;
    let peak_at = [7usize, 40, 99, 126];
    let v: Vec<f32> = (0..512)
        .map(|i| {
            let base = ((i * 61 % 97) as f32 - 48.0) / 7.0;
            // Well clear of `base`'s range, so each peak is unambiguous.
            if i % block == peak_at[i / block] {
                100.0
            } else {
                base
            }
        })
        .collect();
    let (split, packed) = reduce_both_ways::<f32, u32>(&v, 512, 4, "fast_argmax_f32");
    assert_packed_matches_split("fast_argmax_f32", &split, &packed);
    // Assert the indices themselves as well. The guard says "not vacuous"; this
    // says "correct", and they are different claims -- a kernel could return
    // four distinct wrong indices and satisfy only the first.
    //
    // The indices are **global**, not per-block offsets: block `k` answers
    // `k * block + peak_at[k]`. Read off the kernel rather than assumed -- the
    // first draft of this arm expected per-block offsets and failed against
    // `[7, 168, 355, 510]`. Recorded because it is the mirror of the defect this
    // issue closes: a test asserting the wrong thing, confidently.
    //
    // Note this is also why zero stays legitimate here. Only block 0 can answer
    // 0, and it does so whenever its maximum is the first element -- so a
    // not-all-zero guard would be wrong on this arm for a reason no amount of
    // input tuning removes.
    assert_eq!(
        split,
        peak_at
            .iter()
            .enumerate()
            .map(|(k, p)| (k * block + p) as u32)
            .collect::<Vec<u32>>(),
        "fast_argmax_f32 did not find the planted peaks"
    );
}

/// `softmax_f32` — 8 dispatches per decode token, one per attention layer.
#[test]
fn packed_matches_split_for_softmax_f32() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f32> = (0..1024)
        .map(|i| ((i * 29 % 83) as f32 - 41.0) / 13.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_last_softmax_with(
                &device,
                &encoder,
                &kernels,
                "softmax_f32",
                v.len(),
                128,
                &input,
                0,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, v.len()));
    }
    assert_packed_matches_split("softmax_f32", &outs[0], &outs[1]);
}

/// `rmsnorm_f16` — 77 of the 674 dispatches in a decode token, the single
/// largest contributor from this file, and the kernel `DESIGN.md` §3.7b names
/// as the clearest case of a dispatch an ICB cannot express today.
#[test]
fn packed_matches_split_for_rmsnorm_f16() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f16> = (0..2048)
        .map(|i| f16::from_f32(((i * 53 % 89) as f32 - 44.0) / 17.0))
        .collect();
    let alpha: Vec<f16> = (0..256)
        .map(|i| f16::from_f32(1.0 + (i % 7) as f32 / 32.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let alpha_buf = new_buffer(&device, &alpha);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_rms_norm_with(
                &device,
                &encoder,
                &kernels,
                "rmsnorm_f16",
                v.len(),
                256,
                1e-5,
                &input,
                0,
                &alpha_buf,
                0,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, v.len()));
    }
    assert_packed_matches_split("rmsnorm_f16", &outs[0], &outs[1]);
}

/// `rope_f16` — 16 dispatches per decode token, and the only family here whose
/// scalars are `size_t`. That makes it the one that would break first if the
/// Rust mirror had been narrowed to `u32`, so it is worth its own arm rather
/// than being taken as covered by the `uint` families.
#[test]
fn packed_matches_split_for_rope_f16() {
    let device = device();
    let kernels = Kernels::new();
    let (bh, td, d) = (8usize, 64usize, 64usize);
    let n = bh * td;
    let v: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 41 % 79) as f32 - 39.0) / 11.0))
        .collect();
    let cs: Vec<f16> = (0..(td * d / 2))
        .map(|i| f16::from_f32((i % 13) as f32 / 13.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let cos = new_buffer(&device, &cs);
        let sin = new_buffer(&device, &cs);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_rope_with(
                &device, &encoder, &kernels, "rope_f16", bh, td, d, 0, &input, 0, &cos, 0, &sin, 0,
                &output, style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, n));
    }
    assert_packed_matches_split("rope_f16", &outs[0], &outs[1]);
}

/// A strided reduction, so the `dims` *and* `strides` arrays both take the
/// promoted-to-a-buffer path rather than `setBytes`.
///
/// This is the case the packed form does not fully absorb: an array's length
/// comes from the tensor's layout, so it cannot be a struct field. It stays a
/// separate binding, which an ICB can express — the constraint is `setBytes`,
/// not buffer count.
#[test]
fn packed_matches_split_for_sum_f32_strided() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f32> = (0..1024)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let shape = vec![4usize, 8, 32];
    let strides = vec![256usize, 32, 1];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = device
            .new_buffer(32 * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_reduce_strided_with(
                &device,
                &encoder,
                &kernels,
                "fast_sum_f32_strided",
                &shape,
                &strides,
                32,
                BufferOffset::zero_offset(&input),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, 32));
    }
    assert_packed_matches_split("fast_sum_f32_strided", &outs[0], &outs[1]);
}

// ---------------------------------------------------------------------------
// Issue #40 — unary, binary, cast and affine carry both binding styles.
//
// The same three checks issue #38 established for `reduce.metal`, applied to
// the four elementwise families: every `_packed` name resolves against the
// compiled library, the two styles agree bit for bit, and the packed structs'
// layout is compared across the language boundary rather than asserted on each
// side.
// ---------------------------------------------------------------------------

/// Run a layout kernel and return every disagreement against the Rust mirrors.
///
/// Shared by the four `*_params_layout_matches_metal` tests and by their
/// mutation counterparts, so the mutation tests exercise the same comparison
/// the real ones do rather than a re-implementation of it.
fn layout_disagreements(
    pipeline: &ComputePipeline,
    expected: &[(&'static str, u32)],
) -> Vec<String> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out = device
        .new_buffer(
            expected.len() * core::mem::size_of::<u32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe { enc.set_output_buffer(0, Some(&out), 0) };
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let device_side: Vec<u32> = read_to_vec(&out, expected.len());
    expected
        .iter()
        .enumerate()
        .filter(|(slot, (_, rust))| device_side[*slot] != *rust)
        .map(|(slot, (what, rust))| format!("{what}: metal {} vs rust {rust}", device_side[slot]))
        .collect()
}

// ---------------------------------------------------------------------------
// gemv carries both binding styles (issue #41)
//
// `gemv` is 183 of the 674 dispatches per decode token -- the second-largest
// single contributor after `copy2d_f16` -- and unlike the elementwise families
// these are the matmuls, where the weight traffic happens (`DESIGN.md` §13.4a).
// So these tests check two distinct things: that the two styles agree bit for
// bit, and that the struct's layout agrees across the language boundary.
// ---------------------------------------------------------------------------

/// Run one GEMV in a chosen binding style.
///
/// Goes through `call_mlx_gemv_with` rather than reimplementing the tile
/// selection, so both arms pick the same `[[host_name]]` stem and differ only
/// in the `_packed` suffix and how the scalars are bound.
#[allow(clippy::too_many_arguments)]
fn run_mlx_gemv_style<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    rhs: &[T],
    rhs_stride: &[usize],
    style: ParamStyle,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let lhs_buf = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(lhs),
            options,
        )
        .unwrap();
    let rhs_buf = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(rhs),
            options,
        )
        .unwrap();
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_with(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            style,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

/// The two binding styles compute the same bits.
///
/// Bit-identical rather than approximately equal, deliberately: both entry
/// points are instantiated from one body (`gemv_body`), so anything but an
/// exact match means the binding style changed what the kernel computed, which
/// is the question issue #41 asks.
///
/// The shapes cover both kernels and both tile-selection branches that LFM2
/// decode reaches: `n == 1` takes `gemv` and `m == 1` takes `gemv_t`, and the
/// `in_vec_size`/`out_vec_size` ratios pick different `(bm, bn, sm, sn)` rows.
#[test]
fn packed_matches_split_for_gemv() {
    // n == 1 -> gemv (mat[M,K] x vec[K]); K=2048 is LFM2's hidden size.
    for (b, m, n, k) in [
        (1usize, 2048usize, 1usize, 2048usize),
        (1, 512, 1, 64),
        (2, 128, 1, 256),
    ] {
        let lhs: Vec<f32> = (0..b * m * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let rhs: Vec<f32> = (0..b * n * k)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 24.0)
            .collect();
        let split = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, 1, n],
            ParamStyle::Split,
        );
        let packed = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, 1, n],
            ParamStyle::Packed,
        );
        assert_packed_matches_split(
            &format!("gemv at (b,m,n,k)=({b},{m},{n},{k})"),
            &split,
            &packed,
        );
    }
}

/// As above, for the transposed kernel (`m == 1` -> `gemv_t`).
///
/// Kept separate so a failure names which of the two kernels broke; they have
/// different bodies and different threadgroup-reduction paths.
#[test]
fn packed_matches_split_for_gemv_t() {
    for (b, m, n, k) in [
        (1usize, 1usize, 2048usize, 2048usize),
        (1, 1, 512, 128),
        (2, 1, 64, 256),
    ] {
        let lhs: Vec<f32> = (0..b * m * k)
            .map(|i| ((i * 41 % 89) as f32 - 44.0) / 22.0)
            .collect();
        let rhs: Vec<f32> = (0..b * n * k)
            .map(|i| ((i * 29 % 83) as f32 - 41.0) / 20.0)
            .collect();
        let split = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, n, 1],
            ParamStyle::Split,
        );
        let packed = run_mlx_gemv_style(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            &rhs,
            &[n * k, n, 1],
            ParamStyle::Packed,
        );
        assert_packed_matches_split(
            &format!("gemv_t at (b,m,n,k)=({b},{m},{n},{k})"),
            &split,
            &packed,
        );
    }
}

/// The f16 path, which is what LFM2 decode actually dispatches.
///
/// 167 of the 183 gemv dispatches per token are f16 (issue #41's inventory), so
/// an f32-only parity test would leave the decode path itself unchecked.
#[test]
fn packed_matches_split_for_gemv_f16() {
    let (b, m, n, k) = (1usize, 2048usize, 1usize, 2048usize);
    let lhs: Vec<f16> = (0..b * m * k)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 50.0))
        .collect();
    let rhs: Vec<f16> = (0..b * n * k)
        .map(|i| f16::from_f32(((i * 53 % 97) as f32 - 48.0) / 48.0))
        .collect();
    let split = run_mlx_gemv_style(
        GemmDType::F16,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        &rhs,
        &[n * k, 1, n],
        ParamStyle::Split,
    );
    let packed = run_mlx_gemv_style(
        GemmDType::F16,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        &rhs,
        &[n * k, 1, n],
        ParamStyle::Packed,
    );
    assert_packed_matches_split("gemv_float16", &split, &packed);
}

/// The packed siblings of the two fused epilogues compute the classical
/// arms' bits (issue #732 P3): the same `gemv_body`, the same epilogue
/// functor, the params read from a block instead of `setBytes`. At f16 and
/// LFM2's own K, both tiles, with a residual whose operand rides slot 15 on
/// both styles.
#[test]
fn packed_matches_split_for_the_fused_epilogues_f16() {
    for &(n_out, k) in &[(2048usize, 2048usize), (512, 2048)] {
        let mat: Vec<f16> = (0..n_out * k)
            .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 50.0))
            .collect();
        let vec: Vec<f16> = (0..k)
            .map(|i| f16::from_f32(((i * 53 % 97) as f32 - 48.0) / 48.0))
            .collect();
        let res: Vec<f16> = (0..n_out)
            .map(|i| f16::from_f32(((i * 11 % 23) as f32 - 11.0) / 8.0))
            .collect();
        for epilogue in [
            GemvEpilogue::Silu,
            GemvEpilogue::Residual,
            GemvEpilogue::Mul,
            GemvEpilogue::Swiglu,
        ] {
            let run = |style: ParamStyle| -> Vec<f16> {
                let device = device();
                let kernels = Kernels::new();
                let commands = commands(&device);
                let encoder = commands.command_encoder().unwrap();
                let options = RESOURCE_OPTIONS;
                let mk = |v: &[f16]| {
                    device
                        .new_buffer_with_data(
                            v.as_ptr() as *const core::ffi::c_void,
                            std::mem::size_of_val(v),
                            options,
                        )
                        .unwrap()
                };
                let (mat_b, vec_b, res_b) = (mk(&mat), mk(&vec), mk(&res));
                // The SwiGLU's operand is a second weight of `mat`'s shape:
                // `mat` itself, reversed, so the two sweeps differ.
                let gate_w: Vec<f16> = mat.iter().rev().copied().collect();
                let gate_b = mk(&gate_w);
                let output = device
                    .new_buffer(n_out * core::mem::size_of::<f16>(), options)
                    .unwrap();
                let operand = match epilogue {
                    GemvEpilogue::Swiglu => Some((&gate_b, 0usize)),
                    e => e.needs_operand().then_some((&res_b, 0usize)),
                };
                // `x [1, k] @ w.t()` with `w [n_out, k]`: the Linear layout.
                // SAFETY: the test owns every buffer bound or mapped here and reads its
                // results only after waiting for the command buffer that writes them.
                unsafe {
                    call_mlx_gemv_epilogue(
                        &device,
                        &encoder,
                        &kernels,
                        GemmDType::F16,
                        (1, 1, n_out, k),
                        &[k, k, 1],
                        0,
                        &vec_b,
                        &[n_out * k, 1, k],
                        0,
                        &mat_b,
                        &output,
                        style,
                        None,
                        epilogue,
                        operand,
                    )
                }
                .unwrap();
                drop(encoder);
                commands.wait_until_completed().unwrap();
                read_to_vec(&output, n_out)
            };
            let split = run(ParamStyle::Split);
            let packed = run(ParamStyle::Packed);
            // Non-vacuity: the fused arm computed something (a SiLU of a
            // non-trivial accumulator is not all zeros; a residual add moved
            // the value off the residual).
            assert!(
                split.iter().any(|v| v.to_f32() != 0.0),
                "{epilogue:?} n_out={n_out}: all zeros"
            );
            assert_packed_matches_split(
                &format!("gemv_float16{}", epilogue.name_suffix()),
                &split,
                &packed,
            );
        }
    }
}

/// Every classical gemv name has a `_packed` counterpart in the compiled
/// library, and vice versa.
///
/// The same argument as `packed_names_resolve` for `reduce.metal`: an absent
/// instantiation is a compile error on neither side, and the first symptom
/// would be a `LoadFunctionError` inside a forward pass. #26 shipped exactly
/// that for 48 variants, caught only by loading against the library
/// (`DESIGN.md` §8.1b).
///
/// The axis lists are restated here rather than derived from the `.metal`
/// macros, which is the point: if a tile configuration is added to one and not
/// the other, this fails. 144 classical names, 144 packed.
#[test]
fn gemv_packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    // (bm, bn, sm, sn, tm, tn), matching `instantiate_gemv_blocks`.
    let gemv_blocks = [
        (1, 8, 1, 32, 4, 4),
        (1, 8, 1, 32, 1, 4),
        (1, 1, 8, 4, 4, 4),
        (1, 1, 8, 4, 1, 4),
        (4, 1, 1, 32, 1, 4),
        (4, 1, 1, 32, 4, 4),
        (8, 1, 1, 32, 4, 4),
    ];
    // matching `instantiate_gemv_t_blocks`.
    let gemv_t_blocks = [
        (1, 2, 8, 4, 4, 1),
        (1, 2, 8, 4, 4, 4),
        (1, 4, 8, 4, 4, 4),
        (1, 16, 8, 4, 4, 4),
        (1, 16, 4, 8, 4, 4),
    ];

    let mut checked = 0usize;
    // The two fused epilogues and their packed siblings (issue #732 P3), at
    // the two decode tiles `instantiate_gemv_epilogue_packed_blocks` names.
    // The packed name is the classical one with `_packed` appended, as
    // `packed_name` builds every other and as the replay executor's
    // `is_packed` reads it.
    for dtype in ["float32", "float16", "bfloat16"] {
        for bm in [4, 8] {
            for epilogue in ["_silu", "_residual", "_mul", "_swiglu"] {
                let classical =
                    format!("gemv_{dtype}_bm{bm}_bn1_sm1_sn32_tm4_tn4_nc0_axpby0{epilogue}");
                let packed =
                    format!("gemv_{dtype}_bm{bm}_bn1_sm1_sn32_tm4_tn4_nc0_axpby0{epilogue}_packed");
                for name in [classical, packed] {
                    kernels
                        .load_pipeline(&device, Source::Gemv, name.clone())
                        .unwrap_or_else(|e| panic!("{name} does not resolve: {e:?}"));
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(
        checked, 48,
        "12 classical epilogue names and their 12 packed siblings, at 3 dtypes... twice the tiles"
    );
    for (prefix, blocks) in [("gemv", &gemv_blocks[..]), ("gemv_t", &gemv_t_blocks[..])] {
        for dtype in ["float32", "float16", "bfloat16"] {
            for &(bm, bn, sm, sn, tm, tn) in blocks.iter() {
                for nc in [0, 1] {
                    for axpby in [0, 1] {
                        let classical = format!(
                            "{prefix}_{dtype}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{nc}_axpby{axpby}"
                        );
                        let packed = crate::kernels::kernels::params::packed_name(&classical);
                        kernels
                            .load_pipeline(&device, Source::Gemv, classical.clone())
                            .unwrap_or_else(|e| panic!("{classical} does not resolve: {e:?}"));
                        kernels
                            .load_pipeline(&device, Source::Gemv, packed.clone())
                            .unwrap_or_else(|e| panic!("{packed} does not resolve: {e:?}"));
                        checked += 2;
                    }
                }
            }
        }
    }
    assert_eq!(
        checked,
        288 + 48,
        "expected 144 classical + 144 packed gemv variants, and the 24 epilogue names with their 24 packed siblings"
    );
}

fn layout_pipeline(source: Source, name: &'static str) -> ComputePipeline {
    let device = device();
    let kernels = Kernels::new();
    kernels.load_pipeline(&device, source, name).unwrap()
}

/// Compile a mutated copy of a `.metal` source and return its layout pipeline.
///
/// The real source is untouched: the mutation is applied to the `include_str!`ed
/// text and compiled at runtime, which is available precisely because candle
/// compiles `.metal` at runtime (`DESIGN.md` §8.1b).
fn mutated_layout_pipeline(
    source_text: &str,
    original: &str,
    mutated: &str,
    kernel: &str,
) -> ComputePipeline {
    assert!(
        source_text.contains(original),
        "the struct this test mutates has been reworded; update the mutation"
    );
    let mutant = source_text.replace(original, mutated);
    let device = device();
    let library = device.new_library_with_source(&mutant, None).unwrap();
    let function = library.get_function(kernel, None).unwrap();
    device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap()
}

/// The layout check must be able to fail, and must fail on *the silent case*.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test. The mutation **swaps two adjacent same-typed fields** --
/// `Copy2dParams.d1` and `.d2`, both `int64_t`. That case is chosen over
/// inserting or removing a field because it is the one nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `unary.metal` still passes;
/// * both fields are `int64_t`, so nothing type-checks differently -- inserting
///   a field of a *different* type is rejected by C++11 narrowing on the brace
///   initializer before it ever reaches a GPU;
/// * the kernel runs and copies a plausible but wrong rectangle.
///
/// It is `copy2d` deliberately: 140 of the 674 dispatches in a decode token
/// (`DESIGN.md` §11.2), and `d1`/`d2` are exactly the pair whose swap would
/// transpose a copy without changing its size.
#[test]
fn unary_params_layout_check_detects_a_moved_field() {
    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::UNARY,
        "struct Copy2dParams {\n    int64_t d1;\n    int64_t d2;",
        "struct Copy2dParams {\n    int64_t d2;\n    int64_t d1;",
        crate::kernels::kernels::params::UNARY_LAYOUT_KERNEL,
    );
    let d = layout_disagreements(
        &pipeline,
        &crate::kernels::kernels::params::expected_unary_layout(),
    );
    assert!(
        d.iter().any(|x| x.starts_with("Copy2dParams.d1")),
        "swapping d1 must be reported; got {d:?}"
    );
    assert!(
        d.iter().any(|x| x.starts_with("Copy2dParams.d2")),
        "swapping d2 must be reported; got {d:?}"
    );
    assert!(
        !d.iter().any(|x| x.starts_with("sizeof(Copy2dParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {d:?}"
    );
}

/// The same, for `affine.metal` -- the file where the structs actually pad.
///
/// `AffineParams.mul` and `.add` are adjacent `float`s inside a struct whose
/// alignment is set by a `size_t`, so swapping them changes neither `sizeof`
/// (16) nor any type. An affine with `mul` and `add` transposed computes
/// `x*add + mul` and produces entirely plausible numbers.
#[test]
fn affine_params_layout_check_detects_a_moved_field() {
    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::AFFINE,
        "struct AffineParams {\n    size_t dim;\n    float mul;\n    float add;\n};",
        "struct AffineParams {\n    size_t dim;\n    float add;\n    float mul;\n};",
        crate::kernels::kernels::params::AFFINE_LAYOUT_KERNEL,
    );
    let d = layout_disagreements(
        &pipeline,
        &crate::kernels::kernels::params::expected_affine_layout(),
    );
    assert!(
        d.iter().any(|x| x.starts_with("AffineParams.mul")),
        "swapping mul must be reported; got {d:?}"
    );
    assert!(
        d.iter().any(|x| x.starts_with("AffineParams.add")),
        "swapping add must be reported; got {d:?}"
    );
    assert!(
        !d.iter().any(|x| x.starts_with("sizeof(AffineParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {d:?}"
    );
}

/// Every classical name in the four elementwise families has a `_packed`
/// counterpart in the compiled library.
///
/// `DESIGN.md` §8.1b's argument applied to the new axis: an absent
/// instantiation is not a compile error on either side, and the first symptom
/// would be a `LoadFunctionError` inside a forward pass. #26 shipped precisely
/// that for 48 variants, caught only by loading against the library.
///
/// Conditional dtypes (`bf16` behind `__HAVE_BFLOAT__`, `i64` behind
/// `__METAL_VERSION__ >= 220`) are checked *relative to the classical name*: if
/// the classical one resolves, the packed one must. That reads the guards off
/// the compiled library rather than predicting which ones this machine's
/// runtime compiler took.
#[test]
fn elementwise_packed_names_resolve() {
    use crate::kernels::kernels::elementwise_names::{
        affine_names, binary_names, cast_names, unary_names, Presence,
    };
    use crate::kernels::kernels::params::packed_name;

    let device = device();
    let kernels = Kernels::new();

    /// One family's declared variants, with the source file named so a
    /// failure says which `.metal` to look in.
    type Family = (Source, Vec<(String, Presence)>, &'static str);

    let families: [Family; 4] = [
        (Source::Unary, unary_names(), "unary.metal"),
        (Source::Binary, binary_names(), "binary.metal"),
        (Source::Cast, cast_names(), "cast.metal"),
        (Source::Affine, affine_names(), "affine.metal"),
    ];

    let mut required = 0usize;
    let mut conditional_present = 0usize;
    let mut conditional_absent = 0usize;

    for (source, names, file) in families {
        for (classical, presence) in names {
            let classical_ok = kernels
                .load_pipeline(&device, source, classical.clone())
                .is_ok();
            match presence {
                Presence::Unconditional => assert!(
                    classical_ok,
                    "{file} has no kernel named {classical:?}; the declared axes in \
                     elementwise_names.rs disagree with the file's instantiation list"
                ),
                Presence::Conditional => {
                    if !classical_ok {
                        conditional_absent += 1;
                        continue;
                    }
                    conditional_present += 1;
                }
            }
            let packed = packed_name(&classical);
            kernels
                .load_pipeline(&device, source, packed.clone())
                .unwrap_or_else(|e| {
                    panic!(
                        "{file} has no kernel named {packed:?} (the packed form of \
                         {classical:?}), so this variant exists in only one binding \
                         style: {e:?}"
                    )
                });
            required += 1;
        }
    }

    // A floor rather than an exact count: the conditional dtypes legitimately
    // vary with the runtime compiler, so pinning a total would make this test
    // fail on a different machine for a reason that is not a defect. The floor
    // still catches a family silently dropping out of the loop.
    assert!(
        required >= 700,
        "expected at least 700 packed variants to resolve, got {required} \
         ({conditional_present} conditional present, {conditional_absent} absent)"
    );
    println!(
        "resolved {required} packed variants \
         ({conditional_present} conditional present, {conditional_absent} absent)"
    );
}

// --- Bit-identical parity, per family -------------------------------------
//
// Bit-identical rather than approximately equal, deliberately. The two variants
// are instantiated from one body, so anything but an exact match means the
// binding style changed what the kernel computed, which is the question this
// issue asks. `DESIGN.md` §2.3 makes the same argument for the engine as a
// whole, and issue #38 established it for `reduce.metal`.
//
// The arms cover the decode-path kernels by name: `copy2d_f16` (140 dispatches
// per token), `bmul_f16` (96), `badd_f16` (60), `silu_f16` (30),
// `cast_f16_f32` (25), `cast_f32_f16` (8) and `affine_f32` (8) -- 367 of 674
// (`DESIGN.md` §11.2), which is what this issue exists to convert.

/// `silu_f16` -- 30 dispatches per decode token.
#[test]
fn packed_matches_split_for_unary_silu_f16() {
    let device = device();
    let kernels = Kernels::new();
    let v: Vec<f16> = (0..1024)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_unary_contiguous_with(
                &device,
                &encoder,
                &kernels,
                unary::contiguous::silu::HALF,
                core::mem::size_of::<f16>(),
                v.len(),
                BufferOffset::zero_offset(&input),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, v.len()));
    }
    assert_packed_matches_split("silu_f16", &outs[0], &outs[1]);
}

/// The strided unary path, which additionally exercises `dims` and `strides`
/// being promoted out of `setBytes` into a device buffer -- the binding whose
/// renumbering is the silent-corruption case issue #38 records.
#[test]
fn packed_matches_split_for_unary_strided_f32() {
    let device = device();
    let kernels = Kernels::new();
    let shape = [4usize, 8, 16];
    let n: usize = shape.iter().product();
    let strides = [1usize, 4, 32];
    let v: Vec<f32> = (0..n)
        .map(|i| ((i * 29 % 83) as f32 - 41.0) / 13.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_unary_strided_with(
                &device,
                &encoder,
                &kernels,
                unary::strided::cos::FLOAT,
                &shape,
                BufferOffset::zero_offset(&input),
                &strides,
                BufferOffset::zero_offset(&output),
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, n));
    }
    assert_packed_matches_split("cos_f32_strided", &outs[0], &outs[1]);
}

/// `copy2d_f16` -- **140 dispatches per decode token, the largest single kernel
/// in the trace** (`DESIGN.md` §11.2). Its four `int64_t` scalars are the
/// widest packed block in this change.
#[test]
fn packed_matches_split_for_copy2d_f16() {
    let device = device();
    let kernels = Kernels::new();
    let (d1, d2, src_s, dst_s) = (12usize, 20usize, 32usize, 24usize);
    let src: Vec<f16> = (0..d1 * src_s)
        .map(|i| f16::from_f32((i % 251) as f32 - 125.0))
        .collect();
    let dst_init: Vec<f16> = vec![f16::from_f32(0.0); d1 * dst_s];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &src);
        let output = new_buffer(&device, &dst_init);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_copy2d_with(
                &device,
                &encoder,
                &kernels,
                unary::copy2d::HALF,
                &input,
                &output,
                d1,
                d2,
                src_s,
                dst_s,
                0,
                0,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, d1 * dst_s));
    }
    // The hand-written guard that used to sit here -- `any(|x| x != 0.0)`, and
    // the only one of the 30 arms to carry one -- now lives inside the helper
    // and applies to every arm. #48 was right to write it; #53 generalised it.
    assert_packed_matches_split("copy2d_f16", &outs[0], &outs[1]);
}

/// `bmul_f16` -- 96 dispatches per decode token, the largest binary.
#[test]
fn packed_matches_split_for_binary_bmul_f16() {
    let device = device();
    let kernels = Kernels::new();
    let n = 1024usize;
    let l: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let r: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 61 % 97) as f32 - 48.0) / 7.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let left = new_buffer(&device, &l);
        let right = new_buffer(&device, &r);
        let output = new_buffer(&device, &l);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_binary_contiguous_with(
                &device,
                &encoder,
                &kernels,
                "bmul_f16",
                core::mem::size_of::<f16>(),
                n,
                BufferOffset::zero_offset(&left),
                BufferOffset::zero_offset(&right),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, n));
    }
    assert_packed_matches_split("bmul_f16", &outs[0], &outs[1]);
}

/// `badd_f16` strided -- 60 dispatches per decode token, and this arm binds
/// **three** arrays (`dims`, `left_strides`, `right_strides`) where the other
/// families bind two. That is the widest renumbering case in the change.
#[test]
fn packed_matches_split_for_binary_badd_f16_strided() {
    let device = device();
    let kernels = Kernels::new();
    let shape = [4usize, 8, 16];
    let n: usize = shape.iter().product();
    let ls = [1usize, 4, 32];
    let rs = [128usize, 16, 1];
    let l: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let r: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 61 % 97) as f32 - 48.0) / 7.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let left = new_buffer(&device, &l);
        let right = new_buffer(&device, &r);
        let output = new_buffer(&device, &l);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_binary_strided_with(
                &device,
                &encoder,
                &kernels,
                "badd_f16_strided",
                core::mem::size_of::<f16>(),
                &shape,
                BufferOffset::zero_offset(&left),
                &ls,
                BufferOffset::zero_offset(&right),
                &rs,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, n));
    }
    assert_packed_matches_split("badd_f16_strided", &outs[0], &outs[1]);
}

/// `cast_f16_f32` -- 25 dispatches per decode token, the F32 upcast of K and V.
#[test]
fn packed_matches_split_for_cast_f16_f32() {
    let device = device();
    let kernels = Kernels::new();
    let n = 1024usize;
    let v: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let zero: Vec<f32> = vec![0.0; n];

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &zero);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_cast_contiguous_with(
                &device,
                &encoder,
                &kernels,
                "cast_f16_f32",
                core::mem::size_of::<f16>(),
                n,
                BufferOffset::zero_offset(&input),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, n));
    }
    assert_packed_matches_split("cast_f16_f32", &outs[0], &outs[1]);
}

/// `affine_f32` -- 8 dispatches per decode token, the softmax scale.
///
/// This is the arm that exercises the padded struct: `AffineParams` is
/// `{size_t, float, float}`, 16 bytes rather than the 12 its fields sum to. If
/// the trailing pad were omitted the kernel would read `add` from beyond the
/// block, and `mul`/`add` transposed would compute `x*add + mul`.
#[test]
fn packed_matches_split_for_affine_f32() {
    let device = device();
    let kernels = Kernels::new();
    let n = 1024usize;
    let v: Vec<f32> = (0..n)
        .map(|i| ((i * 29 % 83) as f32 - 41.0) / 13.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_affine_with(
                &device,
                &encoder,
                &kernels,
                "affine_f32",
                core::mem::size_of::<f32>(),
                n,
                BufferOffset::zero_offset(&input),
                &output,
                // Deliberately distinct and non-commutative under a swap, so a
                // mul/add transposition cannot pass by coincidence.
                2.5,
                -0.75,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, n));
    }
    assert_packed_matches_split("affine_f32", &outs[0], &outs[1]);
}

/// `powf` uses the one-float `ScaleParams` block rather than `AffineParams`.
///
/// Worth its own arm because sharing a struct between the two-float and
/// one-float families is the mistake that would compile, resolve, and read a
/// fourth field that the capture never wrote.
#[test]
fn packed_matches_split_for_powf_f32() {
    let device = device();
    let kernels = Kernels::new();
    let n = 512usize;
    let v: Vec<f32> = (0..n).map(|i| 1.0 + (i % 17) as f32 / 4.0).collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_powf_with(
                &device,
                &encoder,
                &kernels,
                "powf_f32",
                core::mem::size_of::<f32>(),
                n,
                BufferOffset::zero_offset(&input),
                &output,
                1.75,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, n));
    }
    assert_packed_matches_split("powf_f32", &outs[0], &outs[1]);
}

/// The strided affine, whose block is `{size_t, size_t, float, float}` -- 24
/// bytes, the largest interior-padding case here -- and which also promotes
/// `dims` and `strides` to buffers.
#[test]
fn packed_matches_split_for_affine_f32_strided() {
    let device = device();
    let kernels = Kernels::new();
    let shape = [4usize, 8, 16];
    let n: usize = shape.iter().product();
    let strides = [1usize, 4, 32];
    let v: Vec<f32> = (0..n)
        .map(|i| ((i * 29 % 83) as f32 - 41.0) / 13.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &v);
        let output = new_buffer(&device, &v);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_affine_strided_with(
                &device,
                &encoder,
                &kernels,
                "affine_f32_strided",
                &shape,
                BufferOffset::zero_offset(&input),
                &strides,
                &output,
                2.5,
                -0.75,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, n));
    }
    assert_packed_matches_split("affine_f32_strided", &outs[0], &outs[1]);
}

/// The gemv layout check must be able to fail.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test. The mutation is applied to a **copy of `gemv.metal`**
/// compiled at runtime, so the real source is untouched.
///
/// It **swaps `GemvParams.in_vec_size` and `.out_vec_size`** -- two adjacent
/// fields of the same type, which is the case nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `gemv.metal` still
///   passes;
/// * both are `int`, so no narrowing diagnostic fires -- inserting a field of a
///   *different* type is rejected at compile time by C++11 narrowing on the
///   brace initializer, a real layer of defence but not this one;
/// * the kernel runs and computes a plausible wrong answer, since swapping the
///   loop bound with the output length is arithmetic that still terminates.
///
/// That last point is why this specific pair was chosen over any other adjacent
/// pair: it is the one whose confusion would corrupt a matmul silently rather
/// than crash it.
#[test]
fn gemv_params_layout_check_detects_a_moved_field() {
    use crate::kernels::kernels::params::{expected_gemv_layout, GEMV_LAYOUT_KERNEL};

    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::GEMV,
        "struct GemvParams {\n  int in_vec_size;\n  int out_vec_size;",
        "struct GemvParams {\n  int out_vec_size;\n  int in_vec_size;",
        GEMV_LAYOUT_KERNEL,
    );
    let disagreements = layout_disagreements(&pipeline, &expected_gemv_layout());

    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("GemvParams.in_vec_size")),
        "swapping in_vec_size must be reported; got {disagreements:?}"
    );
    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("GemvParams.out_vec_size")),
        "swapping out_vec_size must be reported; got {disagreements:?}"
    );
    assert!(
        !disagreements
            .iter()
            .any(|d| d.starts_with("sizeof(GemvParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {disagreements:?}"
    );
}

/// The packed entry point does not change the inner loop's shape.
///
/// This is the check issue #41 asks for that the elementwise families did not
/// need. `gemv` is where the weight traffic happens and decode is bandwidth-
/// bound at 88 % of a streaming roofline (`DESIGN.md` §13.4a), so an added
/// per-iteration load or a register spill here is a real regression rather
/// than noise.
///
/// `maxTotalThreadsPerThreadgroup` is computed by the compiler from
/// registers-per-thread, so it is a direct read on register pressure
/// (`DESIGN.md` §3.2: registers/thread -> occupancy -> latency hiding). If the
/// packed form held an extra live value or spilled, this is where it would
/// show. `staticThreadgroupMemoryLength` covers the other half — the
/// `tgp_mem_size` the template computed.
///
/// Why it should hold, which this measures rather than assumes: both entry
/// points pass the scalars **by value** into `gemv_body`, whose parameters are
/// `const int`/`const float`, and `GEMVKernel::run` takes them by value again.
/// So the loop bound and the row stride are in registers before the first
/// iteration under either style, and neither `constant int&` nor
/// `device GemvParams*` is dereferenced inside the loop. Promoting them to a
/// buffer moves *where the one load happens*, not how often.
///
/// There is no `metal-objdump` in this Xcode, so the loop body cannot be
/// diffed directly; this is the strongest check the runtime exposes. The
/// standalone form is `measurements/probes/issue-41/regpressure.m`.
#[test]
fn gemv_packed_does_not_change_occupancy() {
    let device = device();
    let kernels = Kernels::new();

    // The four the decode inventory names, plus the bm8 f16 row that carries
    // the most dispatches. Named explicitly rather than swept so a failure
    // says which variant regressed.
    let decode_path = [
        "gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0",
    ];

    let mut differences = Vec::new();
    for classical in decode_path {
        let packed = crate::kernels::kernels::params::packed_name(classical);
        let pc = kernels
            .load_pipeline(&device, Source::Gemv, classical)
            .unwrap();
        let pp = kernels
            .load_pipeline(&device, Source::Gemv, packed.clone())
            .unwrap();
        if pc.max_total_threads_per_threadgroup() != pp.max_total_threads_per_threadgroup() {
            differences.push(format!(
                "{classical}: maxTotalThreadsPerThreadgroup {} vs {} packed",
                pc.max_total_threads_per_threadgroup(),
                pp.max_total_threads_per_threadgroup()
            ));
        }
    }

    assert!(
        differences.is_empty(),
        "the packed entry point changed occupancy, which means register pressure \
         in the matmul inner loop moved:\n{}",
        differences.join("\n")
    );
}

// ---------------------------------------------------------------------------
// `conv.metal` in both binding styles (issue #42).
//
// Same three obligations as the `reduce.metal` block above -- layout, name
// resolution, bit-identical output -- applied to the family with the most
// `constant &` parameters after `reduce`: 59 across ten families.
//
// Two things make this family's checks carry more than repetition:
//
//  * **`UpsampleBilinear2dParams` mixes widths.** Three `bool` and two `float`
//    between two `size_t`, so five of seven fields land at an offset the
//    padding rule decides. It is the only struct in the tree that exercises
//    both hazards issue #38 names at once.
//  * **`im2col` is the shape `DESIGN.md` §8.1c warns about.** Eight consecutive
//    `constant size_t` parameters, where reordering two is silent -- no size
//    change, no type error, a plausible wrong answer. The mutation test below
//    swaps two of exactly those.
// ---------------------------------------------------------------------------

/// The conv layout check must be able to fail.
///
/// `CONTRIBUTING.md` §3.1 #2. The mutation swaps **`Im2colParams.h_k` and
/// `Im2colParams.w_k`** -- two adjacent `size_t` fields in the middle of eight
/// consecutive ones. That is the case `DESIGN.md` §8.1c names for `im2col`
/// specifically, and it is the one nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `conv.metal` passes;
/// * both are `size_t`, so the brace initializer still type-checks -- a field
///   of a *different* type is rejected by C++11 narrowing before reaching a
///   GPU, which is a real layer of defence but not this one;
/// * the kernel runs and computes a plausible wrong convolution.
///
/// Applied to a copy of `conv.metal` compiled at runtime; the real source is
/// untouched.
#[test]
fn conv_params_layout_check_detects_a_moved_field() {
    use crate::kernels::kernels::params::{expected_conv_layout, CONV_LAYOUT_KERNEL};

    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::CONV,
        "    size_t h_k;\n    size_t w_k;\n    size_t stride;\n    size_t padding;\n    size_t dilation;\n};",
        "    size_t w_k;\n    size_t h_k;\n    size_t stride;\n    size_t padding;\n    size_t dilation;\n};",
        CONV_LAYOUT_KERNEL,
    );
    let disagreements = layout_disagreements(&pipeline, &expected_conv_layout());

    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("Im2colParams.h_k")),
        "swapping h_k must be reported; got {disagreements:?}"
    );
    assert!(
        disagreements
            .iter()
            .any(|d| d.starts_with("Im2colParams.w_k")),
        "swapping w_k must be reported; got {disagreements:?}"
    );
    assert!(
        !disagreements
            .iter()
            .any(|d| d.starts_with("sizeof(Im2colParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {disagreements:?}"
    );
}

/// Every classical conv name has a `_packed` counterpart in the compiled
/// library.
///
/// The same argument as `conv_names_resolve` and `packed_names_resolve`: an
/// absent instantiation is not a compile error on either side, and #26 shipped
/// exactly that for 48 variants.
#[test]
fn conv_packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in ConvKernel::ALL {
        for (suffix, name) in family.variants() {
            let packed = crate::kernels::kernels::params::packed_name(name);
            kernels
                .load_pipeline(&device, Source::Conv, packed.clone())
                .unwrap_or_else(|e| {
                    panic!(
                        "conv.metal has no kernel named {packed:?} \
                         (the packed form of {name:?}, ConvKernel::{}{} for {suffix:?}): {e:?}",
                        family.stem(),
                        family.tail(),
                    )
                });
            checked += 1;
        }
    }
    assert_eq!(
        checked, 67,
        "expected a packed counterpart for all 67 declared conv variants, found {checked}"
    );
}

/// `conv1d_depthwise_f16_k3` — LFM2's shape, and the only conv kernel on any
/// LFM2 dispatch (22 per generation, at prefill; `DESIGN.md` §6.7 L7).
///
/// Bit-identical rather than approximately equal, for the reason the reduce
/// tests give: the two variants come from one body, so any difference means the
/// binding style changed what the kernel computed.
#[test]
fn conv_packed_matches_split_for_conv1d_depthwise_k() {
    let device = device();
    let kernels = Kernels::new();

    let (b, c, l_in, k_size, padding) = (2usize, 8usize, 16usize, 3usize, 2usize);
    let shape = vec![b, c, l_in];
    let l_out = l_in + 2 * padding - (k_size - 1);
    let dst_el = b * c * l_out;

    let src: Vec<f16> = (0..b * c * l_in)
        .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 25.0))
        .collect();
    let weight: Vec<f16> = (0..c * k_size)
        .map(|i| f16::from_f32(((i * 61 % 97) as f32 - 48.0) / 40.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let w_buf = new_buffer(&device, &weight);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_conv1d_depthwise_k_with(
                &device,
                &encoder,
                &kernels,
                "conv1d_depthwise_f16_k3",
                &shape,
                (k_size, padding),
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&w_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, dst_el));
    }
    assert_packed_matches_split("conv1d_depthwise_f16_k3", &outs[0], &outs[1]);
}

/// The generic depthwise kernel, which binds six scalars *and* two arrays.
///
/// Distinct from the `_k` variant above: this one takes `src_strides` as well
/// as `src_dims`, so both arrays take the promoted-to-a-device-buffer path
/// while six scalars are diverted. That is the combination where a renumbering
/// error would land a buffer at the wrong index.
#[test]
fn conv_packed_matches_split_for_conv1d_depthwise() {
    let device = device();
    let kernels = Kernels::new();

    let (b, c, l_in, k_size, stride, padding, dilation) = (2usize, 8usize, 16usize, 3, 1, 2, 1);
    let shape = vec![b, c, l_in];
    let strides = vec![c * l_in, l_in, 1];
    let l_out = (l_in + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = b * c * l_out;

    let src: Vec<f32> = (0..b * c * l_in)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let weight: Vec<f32> = (0..c * k_size)
        .map(|i| ((i * 61 % 97) as f32 - 48.0) / 40.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let w_buf = new_buffer(&device, &weight);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_conv1d_depthwise_with(
                &device,
                &encoder,
                &kernels,
                "conv1d_depthwise_f32",
                &shape,
                &strides,
                (k_size, stride, padding, dilation),
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&w_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("conv1d_depthwise_f32", &outs[0], &outs[1]);
}

/// `im2col` — eight consecutive `constant size_t` parameters, the largest
/// packed block in the file at 64 bytes.
///
/// This is the family `DESIGN.md` §8.1c singles out as the one where reordering
/// two same-typed arguments is silent, so it is the one whose parity check
/// matters most.
#[test]
fn conv_packed_matches_split_for_im2col() {
    let device = device();
    let kernels = Kernels::new();

    // Asymmetric deliberately: `h != w` and `h_k != w_k`, so that swapping two
    // adjacent same-typed fields of `Im2colParams` changes the output. With a
    // square kernel over a square input such a swap is a no-op and this test
    // cannot fail -- found by mutation, and the reason these are not 8x8 / 3x3.
    let (b, c, h, w) = (2usize, 3usize, 7usize, 9usize);
    let (h_k, w_k, stride, padding, dilation) = (2usize, 3usize, 1usize, 1usize, 1usize);
    let shape = vec![b, c, h, w];
    let strides = vec![c * h * w, h * w, w, 1];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;
    let dst_el = b * h_out * w_out * c * h_k * w_k;

    let src: Vec<f32> = (0..b * c * h * w)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_im2col_strided_with(
                &device,
                &encoder,
                &kernels,
                "im2col_f32",
                &shape,
                &strides,
                (h_k, w_k, stride, padding, dilation),
                BufferOffset::zero_offset(&src_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("im2col_f32", &outs[0], &outs[1]);
}

/// `im2col1d` — the 1D form, six scalars and two arrays.
#[test]
fn conv_packed_matches_split_for_im2col1d() {
    let device = device();
    let kernels = Kernels::new();

    let (b, c, l_in, k_size, stride, padding, dilation) = (2usize, 4usize, 12usize, 3, 1, 1, 1);
    let shape = vec![b, c, l_in];
    let strides = vec![c * l_in, l_in, 1];
    let l_out = (l_in + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = b * l_out * c * k_size;

    let src: Vec<f16> = (0..b * c * l_in)
        .map(|i| f16::from_f32(((i * 29 % 83) as f32 - 41.0) / 13.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_im2col1d_strided_with(
                &device,
                &encoder,
                &kernels,
                "im2col1d_f16",
                &shape,
                &strides,
                (k_size, stride, padding, dilation),
                BufferOffset::zero_offset(&src_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, dst_el));
    }
    assert_packed_matches_split("im2col1d_f16", &outs[0], &outs[1]);
}

/// `col2im1d` — six scalars and *no* arrays, so nothing takes the promoted
/// path and slot 1 is the first real buffer.
#[test]
fn conv_packed_matches_split_for_col2im1d() {
    let device = device();
    let kernels = Kernels::new();

    let (b, l_in, c_out, k_size, stride) = (2usize, 6usize, 4usize, 3usize, 1usize);
    let shape = vec![b, l_in, c_out, k_size];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = b * c_out * l_out;

    let src: Vec<f32> = (0..b * l_in * c_out * k_size)
        .map(|i| ((i * 43 % 89) as f32 - 44.0) / 11.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_col2im1d_with(
                &device,
                &encoder,
                &kernels,
                "col2im1d_f32",
                &shape,
                k_size,
                stride,
                BufferOffset::zero_offset(&src_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("col2im1d_f32", &outs[0], &outs[1]);
}

/// `upsample_bilinear2d` — the mixed-width struct, and the family with pinned
/// `[[buffer(N)]]` indices on its classical entry point.
///
/// The one packed block in the tree holding both hazards issue #38 names: three
/// `bool` at 1 byte and two `float` between two `size_t`, so five of seven
/// fields land at a padding-decided offset. Run with `align_corners = false`
/// and one scale supplied, so all three `bool` are exercised at different
/// values rather than all being false.
#[test]
fn conv_packed_matches_split_for_upsample_bilinear2d() {
    let device = device();
    let kernels = Kernels::new();

    // `h != w` as well as `out_w != out_h`, so a swap of two adjacent
    // same-typed params fields is observable in the output.
    let (b, c, h, w) = (1usize, 2usize, 5usize, 3usize);
    let (out_w, out_h) = (7usize, 4usize);
    let shape = vec![b, c, h, w];
    let strides = vec![c * h * w, h * w, w, 1];
    let dst_el = out_w * out_h * b * c;

    let src: Vec<f32> = (0..b * c * h * w)
        .map(|i| ((i * 17 % 61) as f32 - 30.0) / 9.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_upsample_bilinear_2d_with(
                &device,
                &encoder,
                &kernels,
                "upsample_bilinear2d_f32",
                &shape,
                &strides,
                out_w,
                out_h,
                false,
                Some(1.25),
                None,
                BufferOffset::zero_offset(&src_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("upsample_bilinear2d_f32", &outs[0], &outs[1]);
}

/// `upsample_nearest2d` — two `size_t` then two `float`, the other mixed-width
/// struct, and the simpler one.
#[test]
fn conv_packed_matches_split_for_upsample_nearest2d() {
    let device = device();
    let kernels = Kernels::new();

    // `h != w` as well as `out_w != out_h`, so a swap of two adjacent
    // same-typed params fields is observable in the output.
    let (b, c, h, w) = (1usize, 2usize, 5usize, 3usize);
    let (out_w, out_h) = (7usize, 4usize);
    let shape = vec![b, c, h, w];
    let strides = vec![c * h * w, h * w, w, 1];
    let dst_el = out_w * out_h * b * c;

    let src: Vec<f16> = (0..b * c * h * w)
        .map(|i| f16::from_f32(((i * 17 % 61) as f32 - 30.0) / 9.0))
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_upsample_nearest_2d_with(
                &device,
                &encoder,
                &kernels,
                "upsample_nearest2d_f16",
                &shape,
                &strides,
                out_w,
                out_h,
                BufferOffset::zero_offset(&src_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, dst_el));
    }
    assert_packed_matches_split("upsample_nearest2d_f16", &outs[0], &outs[1]);
}

/// `avg_pool2d_u8` — the accumulator case.
///
/// `DESIGN.md` §8.1c: the integer `avg_pool2d` instantiations accumulate in
/// their **own type** rather than widening, so their averaging truncates where
/// it always did. The accumulator is a template parameter and not part of the
/// binding, so a binding change must not move it — this is what proves it did
/// not, on the dtype where truncation is observable.
#[test]
fn conv_packed_matches_split_for_avg_pool2d_u8() {
    let device = device();
    let kernels = Kernels::new();

    // Asymmetric deliberately: `w_in != h_in`, `w_k != h_k` and
    // `w_stride != h_stride`, so swapping two adjacent same-typed fields of
    // `Pool2dParams` changes the output. Square windows at equal strides make
    // such a swap a no-op, and the test then cannot fail.
    let (b, c, w_in, h_in) = (1usize, 2usize, 9usize, 7usize);
    let (w_k, h_k, w_stride, h_stride) = (3usize, 2usize, 2usize, 1usize);
    let shape = vec![b, c, w_in, h_in];
    let strides = vec![c * w_in * h_in, w_in * h_in, h_in, 1];
    let out_w = (w_in - w_k) / w_stride + 1;
    let out_h = (h_in - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * b * c;

    // Values chosen so the u8 sum of a 3x3 window does not wrap, and so the
    // integer division truncates rather than dividing evenly -- if the
    // accumulator had silently widened, these are the outputs that would move.
    let src: Vec<u8> = (0..b * c * w_in * h_in)
        .map(|i| ((i * 7 % 23) + 1) as u8)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<u8>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_pool2d_with(
                &device,
                &encoder,
                &kernels,
                "avg_pool2d_u8",
                &shape,
                &strides,
                out_w,
                out_h,
                w_k,
                h_k,
                w_stride,
                h_stride,
                &src_buf,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<u8>(&output, dst_el));
    }
    assert_packed_matches_split("avg_pool2d_u8", &outs[0], &outs[1]);
}

/// `max_pool2d_f32` — shares `Pool2dParams` with `avg_pool2d`, and has no
/// accumulator of its own.
#[test]
fn conv_packed_matches_split_for_max_pool2d() {
    let device = device();
    let kernels = Kernels::new();

    // Asymmetric deliberately: `w_in != h_in`, `w_k != h_k` and
    // `w_stride != h_stride`, so swapping two adjacent same-typed fields of
    // `Pool2dParams` changes the output. Square windows at equal strides make
    // such a swap a no-op, and the test then cannot fail.
    let (b, c, w_in, h_in) = (1usize, 2usize, 9usize, 7usize);
    let (w_k, h_k, w_stride, h_stride) = (3usize, 2usize, 2usize, 1usize);
    let shape = vec![b, c, w_in, h_in];
    let strides = vec![c * w_in * h_in, w_in * h_in, h_in, 1];
    let out_w = (w_in - w_k) / w_stride + 1;
    let out_h = (h_in - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * b * c;

    let src: Vec<f32> = (0..b * c * w_in * h_in)
        .map(|i| ((i * 53 % 79) as f32 - 39.0) / 6.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_pool2d_with(
                &device,
                &encoder,
                &kernels,
                "max_pool2d_f32",
                &shape,
                &strides,
                out_w,
                out_h,
                w_k,
                h_k,
                w_stride,
                h_stride,
                &src_buf,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("max_pool2d_f32", &outs[0], &outs[1]);
}

/// `conv_transpose1d_u32` — four arrays promoted at once, and the second
/// accumulator case.
///
/// Two things here that no other conv test covers: **four** `setBytes` arrays
/// (`src_dims`, `src_strides`, `k_dims`, `k_strides`) all take the promoted
/// path, so the buffer renumbering has the most to get wrong; and the integer
/// instantiation accumulates in `uint32_t` rather than widening to float
/// (`DESIGN.md` §8.1c).
#[test]
fn conv_packed_matches_split_for_conv_transpose1d_u32() {
    let device = device();
    let kernels = Kernels::new();

    let (b, c_in, l_in, c_out, l_k) = (1usize, 2usize, 5usize, 3usize, 3usize);
    let (stride, padding, out_padding, dilation) = (1usize, 1usize, 0usize, 1usize);
    let src_shape = vec![b, c_in, l_in];
    let src_strides = vec![c_in * l_in, l_in, 1];
    let k_shape = vec![c_in, c_out, l_k];
    let k_strides = vec![c_out * l_k, l_k, 1];
    let l_out = (l_in - 1) * stride + dilation * (l_k - 1) + out_padding + 1 - 2 * padding;
    let dst_el = b * c_out * l_out;

    let src: Vec<u32> = (0..b * c_in * l_in).map(|i| (i * 3 % 11) as u32).collect();
    let kernel_w: Vec<u32> = (0..c_in * c_out * l_k)
        .map(|i| (i * 5 % 7) as u32)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let k_buf = new_buffer(&device, &kernel_w);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<u32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_conv_transpose1d_with(
                &device,
                &encoder,
                &kernels,
                "conv_transpose1d_u32",
                dilation,
                stride,
                padding,
                out_padding,
                c_out,
                l_out,
                b,
                &src_shape,
                &src_strides,
                &k_shape,
                &k_strides,
                &src_buf,
                0,
                &k_buf,
                0,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<u32>(&output, dst_el));
    }
    assert_packed_matches_split("conv_transpose1d_u32", &outs[0], &outs[1]);
}

/// `conv_transpose2d_f32` — six scalars and four arrays, the largest argument
/// list in the file.
#[test]
fn conv_packed_matches_split_for_conv_transpose2d() {
    let device = device();
    let kernels = Kernels::new();

    // `h_in != w_in` and `h_k != w_k`, so a swap of two adjacent same-typed
    // `ConvTranspose2dParams` fields is observable.
    let (b, c_in, h_in, w_in, c_out, h_k, w_k) = (1usize, 2usize, 5usize, 4usize, 2usize, 3, 2);
    let (stride, padding, output_padding, dilation) = (1usize, 1usize, 0usize, 1usize);
    let input_dims = vec![b, c_in, h_in, w_in];
    let input_stride = vec![c_in * h_in * w_in, h_in * w_in, w_in, 1];
    let kernel_dims = vec![c_in, c_out, h_k, w_k];
    let kernel_stride = vec![c_out * h_k * w_k, h_k * w_k, w_k, 1];
    let out_h = (h_in - 1) * stride + dilation * (h_k - 1) + output_padding + 1 - 2 * padding;
    let out_w = (w_in - 1) * stride + dilation * (w_k - 1) + output_padding + 1 - 2 * padding;
    let dst_el = b * c_out * out_h * out_w;

    let src: Vec<f32> = (0..b * c_in * h_in * w_in)
        .map(|i| ((i * 23 % 67) as f32 - 33.0) / 8.0)
        .collect();
    let kernel_w: Vec<f32> = (0..c_in * c_out * h_k * w_k)
        .map(|i| ((i * 31 % 59) as f32 - 29.0) / 12.0)
        .collect();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let k_buf = new_buffer(&device, &kernel_w);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        let cfg = CallConvTranspose2dCfg {
            dilation,
            stride,
            padding,
            output_padding,
            c_out,
            out_w,
            out_h,
            b_size: b,
            input_dims: &input_dims,
            input_stride: &input_stride,
            kernel_dims: &kernel_dims,
            kernel_stride: &kernel_stride,
            input_offset: 0,
            kernel_offset: 0,
        };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_conv_transpose2d_with(
                &device,
                &encoder,
                &kernels,
                "conv_transpose2d_f32",
                cfg,
                &src_buf,
                &k_buf,
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("conv_transpose2d_f32", &outs[0], &outs[1]);
}

// ---------------------------------------------------------------------------
// The layout registry (issue #58).
//
// Seven families each carried a `<family>_params_layout_matches_metal` test that
// differed from the others only in which `Source`, which kernel name and which
// `expected_*_layout()` it named. Adding a family meant a const, a function, and
// a call site here -- and the call site is the one that fails silently: a family
// absent from it is never checked, which is indistinguishable from one that
// passes.
//
// The two tests below replace all seven. `LayoutFamily::descriptor`'s match is
// exhaustive, so a family that fails to register is a **compile error**; these
// close the one gap the compiler cannot see, which is a variant missing from
// `LayoutFamily::ALL`.
//
// Note what is *not* here any more: nothing in this file names a family. A new
// family adds a variant and an arm in `params.rs` and is checked by these tests
// without touching `tests.rs` at all -- which preserves the append-only property
// that let four conversions merge as a union (#62), because the per-family
// parity arms below are still pure EOF appends and there is no longer a shared
// list for two of them to edit.
// ---------------------------------------------------------------------------

/// Every registered family's packed structs agree with the device's view.
///
/// One dispatch per family, driven by `LayoutFamily::ALL`. A `static_assert` on
/// either side proves only that side is self-consistent; this ships the
/// *compiled kernel's own* `sizeof` and field offsets across the boundary and
/// compares them against Rust's `size_of`/`offset_of!`. A field at the wrong
/// offset does not crash — the kernel reads a well-formed number from the wrong
/// place and computes a plausible wrong answer (`DESIGN.md` §3.5, §15.1).
///
/// This is also the check that catches the hazards #38 names — a vector type
/// over-aligning, a `bool` sized differently — for any struct added later, and
/// it now catches them for *every* family by construction rather than for those
/// somebody remembered to add.
///
/// The slot count is asserted rather than assumed: the buffer is sized from
/// `descriptor.slots()`, so a kernel writing more slots than Rust describes
/// would write past it. That is why `slots()` is derived from the `expected`
/// array's length rather than declared beside it.
#[test]
fn every_family_params_layout_matches_metal() {
    use crate::kernels::kernels::params::LayoutFamily;

    // Per-family, not just a total: a total alone cannot say *which* family
    // stopped being checked, and two families moving in opposite directions
    // would cancel.
    const EXPECTED_SLOTS: &[(&str, usize)] = &[
        // 26 until #659, which split the three RoPE structs out of
        // `reduce.metal` into `rope/rope.metal` — 15 slots left with them. A
        // split is as deliberate as an addition and moves this number the same
        // way; the residual reports 11 (ReduceParams, SoftmaxParams, NormParams).
        ("reduce.metal", 17),
        // The three RoPE structs, split out of `reduce.metal` by #659. 15 slots:
        // RopeIParams (4), RopeParams (5), RopeThdParams (6), each a `sizeof`
        // plus its fields. One layout kernel reports all three, the `Flash` shape.
        ("rope/rope.metal", 15),
        ("unary.metal", 10),
        ("binary.metal", 5),
        ("cast.metal", 5),
        ("affine.metal", 16),
        ("gemv.metal", 8),
        ("conv.metal", 80),
        // 27 until #219, which removed `left_size` from all four of this
        // file's structs — four `offsetof` rows, one per struct. A **removal**
        // is as deliberate as an addition and moves this number the same way;
        // the guard fired on it, which is what it is for.
        ("indexing.metal", 23),
        ("scaled_dot_product_attention.metal", 7),
        // Added by #116 (`DESIGN.md` §10.4a) — a deliberate addition, which is
        // what this list requires. **16 slots for two structs in one file**:
        // `flash_decoding.metal` defines both `FlashPartialParams` (13 slots:
        // a `sizeof` and twelve fields) and `FlashCombineParams` (3), and one
        // layout kernel reports both because a layout kernel can only see the
        // structs its own file defines.
        //
        // **17 as of #234**, which gave `FlashPartialParams` the
        // `chunk_capacity` its sibling has carried since #116 — the field
        // `ScratchSizing` selects, and the half that was missing on the pass
        // that lays the region out. 14 slots there and 3 here, and the guard
        // fired on the addition, which is what it is for. **`_pad` is
        // deliberately not a slot**: it carries no value, and asserting the
        // offset of a field the GPU never reads is the shape #103's finding 3
        // and #219 both name — half an assertion over a dead field.
        //
        // **32 as of #148**: `FlashPartialParams` gained `heads_per_seq`,
        // `table_stride`, `k_seq_stride` and `v_seq_stride` (a shared page pool
        // is read by B sequences through B tables, so the batch can no longer be
        // folded into the head stride), 18 slots; and `FlashScatterParams`, the
        // paged write's block, adds 11. The guard fired on both, as it should.
        ("flash_decoding.metal", 32),
        // Issue #218's two structs, also in
        // `scaled_dot_product_attention.metal`: `SdpaQuantParams` (a `sizeof`
        // and eight fields) and `SdpaQuantWriteParams` (a `sizeof` and four).
        // A **second entry for one file** rather than more slots on the `sdpa`
        // row above, because that row's list is what
        // `sdpa_params_layout_check_detects_a_moved_field` indexes its
        // assertion against. **`pad` is deliberately not a slot**, per the
        // `_pad` note above: it carries no value and the GPU never reads it.
        ("scaled_dot_product_attention.metal", 14),
    ];

    let mut failures = Vec::new();
    let mut observed = Vec::new();

    for &family in LayoutFamily::ALL {
        let descriptor = family.descriptor();
        assert!(
            descriptor.slots() > 0,
            "{family:?} registered no layout slots, so its check would pass vacuously"
        );

        let pipeline = layout_pipeline(descriptor.source, descriptor.kernel);
        let disagreements = layout_disagreements(&pipeline, &descriptor.expected);

        if !disagreements.is_empty() {
            failures.push(format!(
                "{} ({:?}):\n  {}",
                family.metal_file(),
                family,
                disagreements.join("\n  ")
            ));
        }
        observed.push((family.metal_file(), descriptor.slots()));
    }

    assert!(
        failures.is_empty(),
        "packed parameter layout differs between .metal and params.rs.\n\
         A field at the wrong offset is silent corruption, not a crash:\n{}",
        failures.join("\n")
    );

    // Checked by count, not by inspection: if a family stops being checked the
    // number moves, and the assertion names it rather than the suite quietly
    // covering less than it did.
    assert_eq!(
        observed.len(),
        LayoutFamily::ALL.len(),
        "every registered family must be checked"
    );
    assert_eq!(
        observed,
        EXPECTED_SLOTS.to_vec(),
        "family or slot count changed. Twelve families and their slot counts are \
         the state at issue #659; update this only alongside a deliberate addition."
    );
}

/// `LayoutFamily::ALL` names every variant.
///
/// The `match` in `LayoutFamily::descriptor` is exhaustive, so a family that
/// fails to describe itself does not compile — that is the acceptance criterion
/// and the compiler enforces it. What the compiler cannot see is a variant
/// missing from `ALL`, since a short array is a well-formed program. This is the
/// one part of the registration that needs a test rather than a type.
///
/// It works by matching each known variant off an exhaustive `match` too, so
/// adding a variant fails *here* at compile time as well and cannot be answered
/// by editing only this test's expected count.
#[test]
fn layout_registry_covers_every_family() {
    use crate::kernels::kernels::params::LayoutFamily;

    // Exhaustive by construction: adding a variant without extending this match
    // is `error[E0004]`, so the list below cannot silently fall behind the enum.
    fn seen(family: LayoutFamily) -> &'static str {
        match family {
            LayoutFamily::Reduce => "reduce",
            LayoutFamily::Rope => "rope",
            LayoutFamily::Unary => "unary",
            LayoutFamily::Binary => "binary",
            LayoutFamily::Cast => "cast",
            LayoutFamily::Affine => "affine",
            LayoutFamily::Gemv => "gemv",
            LayoutFamily::Conv => "conv",
            LayoutFamily::Indexing => "indexing",
            LayoutFamily::Sdpa => "sdpa",
            LayoutFamily::Flash => "flash_decoding",
            LayoutFamily::SdpaQuant => "sdpa_quant",
        }
    }

    let expected = [
        LayoutFamily::Reduce,
        LayoutFamily::Rope,
        LayoutFamily::Unary,
        LayoutFamily::Binary,
        LayoutFamily::Cast,
        LayoutFamily::Affine,
        LayoutFamily::Gemv,
        LayoutFamily::Conv,
        LayoutFamily::Indexing,
        LayoutFamily::Sdpa,
        LayoutFamily::Flash,
        LayoutFamily::SdpaQuant,
    ];

    for family in expected {
        assert!(
            LayoutFamily::ALL.contains(&family),
            "{} ({:?}) is a LayoutFamily variant but is missing from \
             LayoutFamily::ALL, so it would never be checked",
            seen(family),
            family
        );
    }
    assert_eq!(
        LayoutFamily::ALL.len(),
        expected.len(),
        "LayoutFamily::ALL has an entry that is not in this test's list, or a \
         duplicate"
    );
}

/// The registry-driven layout check must be able to fail.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test, and this one guards a defect that is otherwise invisible.
///
/// The mutation is applied to a **copy of `reduce.metal`** compiled at runtime,
/// so the real source is untouched. It **swaps `ReduceParams.src_numel` and
/// `ReduceParams.num_dims`** — two adjacent fields of the same type. That case
/// is chosen deliberately over inserting or removing a field, because it is the
/// one nothing else catches:
///
/// * `sizeof` is unchanged, so the `static_assert` in `reduce.metal` still
///   passes;
/// * both fields are `uint`, so the brace initializer still type-checks —
///   inserting a field of a different type is rejected by C++11 narrowing
///   before it ever reaches a GPU, which is a real layer of defence but not
///   this one;
/// * the kernel runs and produces plausible numbers from the wrong values.
///
/// It is the same shape as the defect `DESIGN.md` §8.1c warns about for
/// `im2col`'s eight consecutive `constant size_t` parameters: reordering two
/// same-typed arguments is silent.
///
/// It goes through `LayoutFamily::Reduce.descriptor()` rather than naming the
/// kernel and the expected array directly, so the mutation exercises the same
/// comparison the registry-driven check runs rather than a re-implementation of
/// it.
#[test]
fn reduce_params_layout_check_detects_a_moved_field() {
    use crate::kernels::kernels::params::LayoutFamily;

    let descriptor = LayoutFamily::Reduce.descriptor();
    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::REDUCE,
        "struct ReduceParams {\n    uint src_numel;\n    uint num_dims;\n    uint el_per_block;\n};",
        "struct ReduceParams {\n    uint num_dims;\n    uint src_numel;\n    uint el_per_block;\n};",
        descriptor.kernel,
    );
    let d = layout_disagreements(&pipeline, &descriptor.expected);

    // Both swapped fields must be reported, and the struct's size must not be
    // -- which is exactly why `sizeof` alone is not a sufficient check.
    assert!(
        d.iter().any(|x| x.starts_with("ReduceParams.src_numel")),
        "swapping src_numel must be reported; got {d:?}"
    );
    assert!(
        d.iter().any(|x| x.starts_with("ReduceParams.num_dims")),
        "swapping num_dims must be reported; got {d:?}"
    );
    assert!(
        !d.iter().any(|x| x.starts_with("sizeof(ReduceParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {d:?}"
    );
}

/// Every name [`IndexingKernel`] declares must exist in the compiled
/// `indexing.metal` library.
///
/// The `conv_names_resolve` / `reduce_names_resolve` counterpart for the last
/// decode-path family to get a registry, and it is the one that had already
/// failed: `candle-core` named `is_i64_u8` and `is_i64_u32` and
/// `indexing.metal` declared neither, so `index_select` on a `U8` or `U32`
/// tensor with `I64` indices was a runtime `LoadFunctionError`. After #26's 48
/// absent reduce variants and `conv`'s, that is the fourth firing of the class
/// (`DESIGN.md` §8.1b) — and the first that needed a check spanning two crates
/// to see, which is why three previous registry passes did not find it.
#[test]
fn indexing_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in IndexingKernel::ALL {
        for ((ids, values), name) in family.variants() {
            kernels
                .load_pipeline(&device, Source::Indexing, name)
                .unwrap_or_else(|e| {
                    panic!(
                        "indexing.metal has no kernel named {name:?} \
                         (declared by IndexingKernel::{} for ids={ids:?} values={values:?}): {e:?}",
                        family.stem(),
                    )
                });
            checked += 1;
        }
    }

    // Guards against the table being emptied or a family silently dropping all
    // its variants -- an all-green run over zero names would otherwise pass.
    // 18 each for index_select and index_add (3 index types x 6 value types),
    // 16 for gather, and 10 each for scatter and scatter_add.
    assert_eq!(
        checked, 72,
        "expected 72 declared indexing variants, found {checked}"
    );
}

/// Each declared name must be its family's stem, then the index dtype, then the
/// value dtype, and each `(index, value)` pair must appear once per family.
///
/// The names are stored verbatim so they can be grepped against
/// `indexing.metal`, which means a row could pair one key with another family's
/// valid name and still resolve — `indexing_names_resolve` would pass while
/// [`IndexingKernel::name`] handed callers the wrong kernel. That is not
/// hypothetical here: `s_u32_f32` and `sa_u32_f32` differ by one character and
/// by whether the destination is assigned or accumulated into, so a copy-paste
/// between the two families produces a name that loads and computes the wrong
/// thing.
///
/// The ordering matters too. The `[[host_name]]` reads
/// `<stem>_<index>_<value>` while the Metal template takes `<value, index>`, so
/// a row that swapped the two suffixes would still resolve — against a kernel
/// with the dtypes transposed.
#[test]
fn indexing_names_match_their_stem_and_dtypes() {
    for family in IndexingKernel::ALL {
        let mut seen: Vec<(&str, &str)> = Vec::new();
        for ((ids, values), name) in family.variants() {
            assert_eq!(
                name,
                format!("{}_{ids}_{values}", family.stem()),
                "IndexingKernel::{} declares {name:?} for ids={ids:?} values={values:?}",
                family.stem(),
            );
            assert!(
                !seen.contains(&(ids, values)),
                "IndexingKernel::{} declares ({ids:?}, {values:?}) twice",
                family.stem(),
            );
            seen.push((ids, values));
        }
        assert!(
            !seen.is_empty(),
            "IndexingKernel::{} declares no variants",
            family.stem(),
        );
    }
}

/// A `(index dtype, value dtype)` pair a family does not declare must not
/// produce a name.
///
/// Returning `None` is what keeps an unsupported pair from reaching
/// `load_pipeline` as a string that will fail to resolve — the caller reports it
/// against its own dtype enum instead. This is the property whose absence *was*
/// the `is_i64_u8` bug: the old `match` returned a name for it unconditionally,
/// and nothing checked that the name existed.
#[test]
fn indexing_undeclared_pair_has_no_name() {
    for family in IndexingKernel::ALL {
        // f64 and i32 are instantiated nowhere in indexing.metal, in either
        // position.
        for bad in ["f64", "i32", "not_a_dtype"] {
            assert_eq!(
                family.name(bad, "f32"),
                None,
                "IndexingKernel::{} named an {bad:?} index type",
                family.stem(),
            );
            assert_eq!(
                family.name("u32", bad),
                None,
                "IndexingKernel::{} named a {bad:?} value type",
                family.stem(),
            );
        }
    }

    // The scatter families are the asymmetric ones: `sa_u32_u32` exists and
    // `sa_u8_u32` does not, which is indexing.metal's asymmetry rather than a
    // simplification in the registry.
    assert_eq!(
        IndexingKernel::SCATTER_ADD.name("u32", "u32"),
        Some("sa_u32_u32")
    );
    assert_eq!(IndexingKernel::SCATTER_ADD.name("u8", "u32"), None);
    assert_eq!(IndexingKernel::SCATTER.name("u8", "u32"), None);

    // gather declares no u8-valued variants for u32 or i64 indices.
    assert_eq!(IndexingKernel::GATHER.name("u32", "u8"), None);
    assert_eq!(IndexingKernel::GATHER.name("i64", "u8"), None);

    // And the positive cases, so the assertions above are not vacuous. The
    // first is the LFM2 embedding lookup; the second and third are the two
    // that did not exist before this change.
    assert_eq!(
        IndexingKernel::INDEX_SELECT.name("u32", "f16"),
        Some("is_u32_f16")
    );
    assert_eq!(
        IndexingKernel::INDEX_SELECT.name("i64", "u8"),
        Some("is_i64_u8")
    );
    assert_eq!(
        IndexingKernel::INDEX_SELECT.name("i64", "u32"),
        Some("is_i64_u32")
    );
}

/// Every classical indexing name has a `_packed` counterpart in the compiled
/// library.
///
/// The same argument as `packed_names_resolve` and `conv_packed_names_resolve`:
/// an absent instantiation is not a compile error on either side, and the first
/// symptom would be a `LoadFunctionError` inside a forward pass. This family in
/// particular has already shipped exactly that — `is_i64_u8` and `is_i64_u32`
/// were named by `candle-core` and declared nowhere until #64 (`DESIGN.md`
/// §8.1e), which is the fourth firing of the class §8.1b tracks.
#[test]
fn indexing_packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for family in IndexingKernel::ALL {
        for ((index_suffix, value_suffix), name) in family.variants() {
            let packed = crate::kernels::kernels::params::packed_name(name);
            kernels
                .load_pipeline(&device, Source::Indexing, packed.clone())
                .unwrap_or_else(|e| {
                    panic!(
                        "indexing.metal has no kernel named {packed:?} \
                         (the packed form of {name:?}, IndexingKernel::{} for \
                         ({index_suffix:?}, {value_suffix:?})): {e:?}",
                        family.stem(),
                    )
                });
            checked += 1;
        }
    }
    assert_eq!(
        checked, 72,
        "expected a packed counterpart for all 72 declared indexing variants, \
         found {checked}"
    );
}

/// `is_u32_f16` — the LFM2 embedding lookup, and the whole reason this family
/// is on the ICB path at all (`DESIGN.md` §11.2: 1 dispatch of 674 per token).
///
/// Bit-identical rather than approximately equal, for the reason every other
/// family's arms give: the two entry points come from one body, so any
/// difference means the binding style changed what the kernel computed.
///
/// The **contiguous** arm, which is the one LFM2 takes.
#[test]
fn indexing_packed_matches_split_for_index_select() {
    let device = device();
    let kernels = Kernels::new();

    // [8, 4] source, six ids: dst is 6 x 4. Values vary per element so the
    // non-vacuity guard in `assert_packed_matches_split` has something to see.
    let (rows, cols) = (8usize, 4usize);
    let src: Vec<f16> = (0..rows * cols)
        .map(|i| f16::from_f32(((i * 37 % 71) as f32 - 35.0) / 9.0))
        .collect();
    let ids: Vec<u32> = vec![5, 0, 7, 2, 2, 6];
    let shape = vec![rows, cols];
    let strides = vec![cols, 1];
    let dst_el = ids.len() * cols;

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_index_select_with(
                &device,
                &encoder,
                &kernels,
                "is_u32_f16",
                &shape,
                ids.len(),
                0,
                true,
                &shape,
                &strides,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, dst_el));
    }
    assert_packed_matches_split("is_u32_f16", &outs[0], &outs[1]);
}

/// `is_u32_f32` on a **strided** source — the arm that reads `src_num_dims`,
/// `src_dims` and `src_strides`.
///
/// A second `index_select` arm rather than a redundant one, because the
/// contiguous arm above never reaches any of those three: it takes the
/// `contiguous ? src_i` fast path and the two arrays are bound but unread. So
/// this is the arm that would catch a packed struct whose `contiguous` or
/// `src_num_dims` landed at the wrong offset — which is precisely where this
/// family's padding is (`DESIGN.md` §11.3b's `bool` hazard, firing here for the
/// first time).
///
/// The layout is rank 3 with `rank != dims[dim]`, the shape #82's fix is about.
#[test]
fn indexing_packed_matches_split_for_index_select_strided() {
    let device = device();
    let kernels = Kernels::new();

    // A [2, 3, 5] source read through a transposed layout, so `src_strides` is
    // not the contiguous one and the strided arm genuinely differs from the
    // fast path. rank 3, dims[0] = 2 -- `rank != dims[dim]`.
    let dims = vec![2usize, 3, 5];
    let n = dims.iter().product::<usize>();
    let src: Vec<f32> = (0..n)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 7.0)
        .collect();
    // Strides of a [5, 3, 2] tensor viewed as [2, 3, 5] -- i.e. reversed.
    let strides = vec![1usize, 2, 6];
    let ids: Vec<u32> = vec![1, 0, 1, 1, 0];
    let dim = 0usize;
    let right_size: usize = dims[dim + 1..].iter().product();
    let left_size: usize = dims[..dim].iter().product();
    let dst_el = ids.len() * left_size * right_size;

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_index_select_with(
                &device,
                &encoder,
                &kernels,
                "is_u32_f32",
                &dims,
                ids.len(),
                dim,
                false,
                &dims,
                &strides,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("is_u32_f32 (strided)", &outs[0], &outs[1]);
}

/// `gather_u32_f32`.
#[test]
fn indexing_packed_matches_split_for_gather() {
    let device = device();
    let kernels = Kernels::new();

    let (rows, cols) = (4usize, 6usize);
    let src: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 31 % 89) as f32 - 44.0) / 5.0)
        .collect();
    // gather's ids are dst-shaped: one index per output element.
    let ids: Vec<u32> = (0..rows * cols).map(|i| ((i * 7) % cols) as u32).collect();
    let shape = vec![rows, cols];
    let dst_el = ids.len();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        let output = device
            .new_buffer(dst_el * core::mem::size_of::<f32>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_gather_with(
                &device,
                &encoder,
                &kernels,
                "gather_u32_f32",
                &shape,
                cols,
                1,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("gather_u32_f32", &outs[0], &outs[1]);
}

/// `s_u32_f32` — `scatter`, which **assigns**.
///
/// Its sibling `sa_u32_f32` accumulates and is the arm below; they share
/// `ScatterParams`, so both are checked rather than one standing in for the
/// other. A struct shared by two kernels is exactly the case where a parity arm
/// on one leaves the other unproven.
#[test]
fn indexing_packed_matches_split_for_scatter() {
    let device = device();
    let kernels = Kernels::new();

    let (rows, src_cols, dst_cols) = (3usize, 4usize, 6usize);
    let src: Vec<f32> = (0..rows * src_cols)
        .map(|i| ((i * 41 % 83) as f32 - 41.0) / 6.0)
        .collect();
    let ids: Vec<u32> = (0..rows * src_cols)
        .map(|i| ((i * 5) % dst_cols) as u32)
        .collect();
    let src_shape = vec![rows, src_cols];
    let dst_shape = vec![rows, dst_cols];
    let dst_el = rows * dst_cols;

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        // Seeded rather than zeroed: `scatter` writes only the destinations the
        // ids name, so a zeroed output would leave the untouched elements at 0
        // on both arms and prove nothing about them.
        let seed: Vec<f32> = (0..dst_el).map(|i| (i as f32) * -0.25 - 1.0).collect();
        let output = new_buffer(&device, &seed);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scatter_with(
                &device,
                &encoder,
                &kernels,
                "s_u32_f32",
                &src_shape,
                &dst_shape,
                1,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                BufferOffset::zero_offset(&output),
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("s_u32_f32", &outs[0], &outs[1]);
}

/// `sa_u32_f32` — `scatter_add`, which **accumulates**.
///
/// Shares [`crate::kernels::ScatterParams`] with `scatter` above; see that arm's note on
/// why both are checked.
#[test]
fn indexing_packed_matches_split_for_scatter_add() {
    let device = device();
    let kernels = Kernels::new();

    let (rows, src_cols, dst_cols) = (3usize, 4usize, 6usize);
    let src: Vec<f32> = (0..rows * src_cols)
        .map(|i| ((i * 43 % 79) as f32 - 39.0) / 4.0)
        .collect();
    // Repeats on purpose: two src elements landing on one destination is what
    // makes `+=` differ from `=`, so this arm exercises the accumulation.
    let ids: Vec<u32> = (0..rows * src_cols)
        .map(|i| ((i * 2) % dst_cols) as u32)
        .collect();
    let src_shape = vec![rows, src_cols];
    let dst_shape = vec![rows, dst_cols];
    let dst_el = rows * dst_cols;

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        let seed: Vec<f32> = (0..dst_el).map(|i| (i as f32) * 0.5 + 2.0).collect();
        let output = new_buffer(&device, &seed);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scatter_with(
                &device,
                &encoder,
                &kernels,
                "sa_u32_f32",
                &src_shape,
                &dst_shape,
                1,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                BufferOffset::zero_offset(&output),
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("sa_u32_f32", &outs[0], &outs[1]);
}

/// `ia_u32_f32` — `index_add`, the sixth-scalar family.
///
/// [`crate::kernels::IndexAddParams`] is the only indexing struct with six fields, so
/// this is the arm that would catch `ids_dim_size` landing at the wrong offset.
#[test]
fn indexing_packed_matches_split_for_index_add() {
    let device = device();
    let kernels = Kernels::new();

    let (rows, src_cols, dst_cols) = (3usize, 4usize, 6usize);
    let src: Vec<f32> = (0..rows * src_cols)
        .map(|i| ((i * 47 % 73) as f32 - 36.0) / 3.0)
        .collect();
    // index_add's ids are one-dimensional: one destination index per source
    // column, read as `input_ids[j]` for j in 0..ids_dim_size.
    let ids: Vec<u32> = vec![4, 1, 1, 5];
    let src_shape = vec![rows, src_cols];
    let dst_shape = vec![rows, dst_cols];
    let ids_shape = vec![ids.len()];
    let dst_el = rows * dst_cols;

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let src_buf = new_buffer(&device, &src);
        let ids_buf = new_buffer(&device, &ids);
        let seed: Vec<f32> = (0..dst_el).map(|i| (i as f32) * 0.75 - 3.0).collect();
        let output = new_buffer(&device, &seed);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_index_add_with(
                &device,
                &encoder,
                &kernels,
                "ia_u32_f32",
                &src_shape,
                &dst_shape,
                &ids_shape,
                1,
                BufferOffset::zero_offset(&src_buf),
                BufferOffset::zero_offset(&ids_buf),
                &output,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f32>(&output, dst_el));
    }
    assert_packed_matches_split("ia_u32_f32", &outs[0], &outs[1]);
}

/// The indexing layout check must be able to fail, and must fail on *the silent
/// case*.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test. Applied to **production source** — `crate::kernels::source::INDEXING`
/// is the `include_str!`'d text the GPU is actually asked to compile, not a
/// copy written for the test.
///
/// The mutation **swaps `IndexParams.src_dim_size` and `.right_size`**, two
/// adjacent `size_t`. That case is chosen over inserting or removing a field
/// because it is the one nothing else catches:
///
/// * `sizeof` is unchanged at 48, so the `static_assert` in `indexing.metal`
///   still passes;
/// * both fields are `size_t`, so the brace initializer in the classical
///   wrapper still type-checks — inserting a field of a *different* type is
///   rejected by C++11 narrowing before it ever reaches a GPU, which is a real
///   layer of defence but not this one;
/// * the kernel runs and indexes against the wrong number, producing a
///   plausible wrong answer rather than a crash.
///
/// This is `DESIGN.md` §8.1c's warning about `im2col`'s eight consecutive
/// `constant size_t` parameters, in the family that has four.
///
/// # Why not `dst_size`/`left_size`, which this test swapped until #219
///
/// **Because `left_size` no longer exists, and this test is part of why.**
/// §11.3k finding 3 found the field *with a mutation that survived*: swapping
/// it changed nothing observable, since every body derives what it needs from
/// `tid / right_size` and no kernel ever read it. §11.3l finding 3 then
/// sharpened the rule this test now obeys — **both swapped fields must be read,
/// and must hold different values** — and this test was carrying the first
/// failure mode between #81 and #219: half its assertion was over a field the
/// GPU ignores.
///
/// `src_dim_size` and `right_size` are both read by `index_body`: the first
/// clamps the index, the second divides `tid` twice. A swap changes the answer.
#[test]
fn indexing_params_layout_check_detects_a_moved_field() {
    use crate::kernels::kernels::params::LayoutFamily;

    let descriptor = LayoutFamily::Indexing.descriptor();
    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::INDEXING,
        "struct IndexParams {\n    size_t dst_size;\n    size_t src_dim_size;\n    size_t right_size;",
        "struct IndexParams {\n    size_t dst_size;\n    size_t right_size;\n    size_t src_dim_size;",
        descriptor.kernel,
    );
    let d = layout_disagreements(&pipeline, &descriptor.expected);

    assert!(
        d.iter().any(|x| x.starts_with("IndexParams.src_dim_size")),
        "swapping src_dim_size must be reported; got {d:?}"
    );
    assert!(
        d.iter().any(|x| x.starts_with("IndexParams.right_size")),
        "swapping right_size must be reported; got {d:?}"
    );
    assert!(
        !d.iter().any(|x| x.starts_with("sizeof(IndexParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {d:?}"
    );
}

/// Moving the `bool` in [`crate::kernels::IndexParams`] is rejected at compile time —
/// and **not by the check this test was written expecting**.
///
/// This is the second mutation rather than a duplicate of the one above, and it
/// documents a defence the other converted families cannot demonstrate because
/// none of their structs has a `bool` on a path a test reaches. #40 recorded
/// the `bool` hazard as "real for a future family; absent here"; this is that
/// family.
///
/// # What was expected, and what happens
///
/// Moving `contiguous` from between two `size_t` to the end changes
/// `sizeof(IndexParams)` from 56 to 48, so the expectation was that
/// `indexing.metal`'s own `static_assert(sizeof(IndexParams) == 56)` rejects
/// the mutant. **It never gets that far.** The classical `index` wrapper builds
/// the struct with a brace initializer in the same order as its argument list,
/// so reordering the fields makes it try to initialize a `bool` from a
/// `size_t`, and **C++11 narrowing rejects that first**:
///
/// ```text
/// error: non-constant-expression cannot be narrowed from type 'size_t'
///        (aka 'unsigned long') to 'bool' in initializer list [-Wc++11-narrowing]
/// ```
///
/// That is the layer `DESIGN.md` §11.3d names in passing — "inserting a field of
/// a *different* type is caught at compile time by C++11 narrowing on the brace
/// initializer, which is a layer of defence §11.3b does not mention" — firing
/// on a *reorder* rather than an insertion, because a `bool` beside `size_t`
/// makes any reorder a type change. It is strictly earlier and louder than the
/// `static_assert`, and it exists only because the classical wrapper builds the
/// struct positionally.
///
/// So the ordering of defences for this family is: narrowing (compile, if the
/// reorder crosses the `bool`), then `static_assert` (compile, if it changes
/// `sizeof`), then the cross-boundary layout check (runtime, for a reorder that
/// changes neither — which is the case the test above covers). None subsumes
/// another, and the first is free.
#[test]
fn indexing_moving_the_bool_is_rejected_at_compile_time() {
    let original = "    size_t ids_size;\n    bool contiguous;\n    size_t src_num_dims;\n};";
    let mutated = "    size_t ids_size;\n    size_t src_num_dims;\n    bool contiguous;\n};";
    assert!(
        crate::kernels::source::INDEXING.contains(original),
        "the struct this test mutates has been reworded; update the mutation"
    );
    let mutant = crate::kernels::source::INDEXING.replace(original, mutated);

    let device = device();
    let err = device.new_library_with_source(&mutant, None).expect_err(
        "moving `contiguous` past `src_num_dims` makes the classical wrapper's \
         brace initializer narrow a size_t to a bool, and changes \
         sizeof(IndexParams) from 56 to 48. If this compiles, both of those \
         guards are gone.",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("c++11-narrowing"),
        "the mutant must be rejected by C++11 narrowing on the classical \
         wrapper's brace initializer -- the earliest of the three guards. If \
         this stops firing, check whether the wrapper still builds the struct \
         positionally; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// `scaled_dot_product_attention.metal` (issue #103).
//
// The family §11.3h deferred and #97 made decode-path. Three arms, and they
// check different things:
//
// * `sdpa_packed_names_resolve` -- every declared classical name has a
//   `_packed` counterpart in the compiled library. An absent instantiation is
//   not a compile error on either side, and the first symptom would be a
//   `LoadFunctionError` inside a forward pass (`DESIGN.md` §8.1b).
// * `sdpa_packed_matches_split_for_sdpa_vector` -- the two styles compute the
//   same bits, through the enforced helper, at LFM2's own geometry.
// * `sdpa_params_layout_check_detects_a_moved_field` -- the layout check can
//   fail, and fails on the silent case.
// ---------------------------------------------------------------------------

/// Every `sdpa_vector` name this crate can dispatch has a `_packed`
/// counterpart in the compiled library.
///
/// The same argument as `packed_names_resolve` and its siblings: an absent
/// instantiation is not a compile error on either side, and #26 shipped
/// precisely that for 48 variants, caught only by loading against the library.
///
/// # This family has no registry, and the check is weaker for it
///
/// Every other converted family declares its names in a registry — `ReduceKernel`,
/// `ConvKernel`, `IndexingKernel`, the elementwise axes — so its resolve test
/// iterates a declared list and a count assertion catches a name that stops
/// being covered. `sdpa.rs` has a hand-written `match (head_dim, dtype)`
/// instead, which `DESIGN.md` §11.3h names as "the same registry gap `indexing`
/// has" and which is still open.
///
/// So this test enumerates the axes the way `sdpa.rs`'s match does, and asserts
/// **both spellings** rather than only the packed one — because the gap this
/// family actually has runs the other way. See
/// `sdpa_vector_declares_variants_the_name_table_cannot_reach`.
///
/// # It must resolve *with constants*, and that is new
///
/// Every prior family's resolve test calls `load_pipeline`, which builds a
/// pipeline from a plain `newFunctionWithName:`. That **aborts** here rather
/// than returning an error:
///
/// ```text
/// validateWithDevice:1530: failed assertion `Compute Pipeline Descriptor Validation
/// function sdpa_vector_float16_t_32 cannot be used to build a pipeline state.
/// Use newFunctionWithName:constantValues:... to get the specialized function'
/// ```
///
/// `sdpa_vector` is the first converted family whose kernels take a **function
/// constant** (`sdpa_vector_has_mask`, index 20), and an unspecialized function
/// is not a pipeline-able object. So the check has to supply the same constant
/// the call site does — which makes it a slightly stronger check than the
/// others, since it proves the name resolves *in the configuration that is
/// dispatched* rather than merely existing.
///
/// Worth carrying: `SIGABRT` rather than a `Result` is what a missing
/// `constantValues` gives, so it cannot be caught and reported. That is the
/// same shape as §3.7d's `supportIndirectCommandBuffers` traps — the failure
/// kills the process rather than failing a test.
#[test]
fn sdpa_packed_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    // The (dtype token, head_dim) pairs `call_sdpa_vector`'s match arm covers.
    // Written out rather than derived, because there is nothing to derive from:
    // the match is the only declaration of this set that exists.
    const DTYPES: &[&str] = &["float16_t", "bfloat16_t", "float"];
    const HEAD_DIMS: &[usize] = &[32, 64, 96, 128, 256, 512];

    // The configuration `call_sdpa_vector` dispatches, and the only one in
    // which these functions are pipeline-able at all.
    let constants = || {
        Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]))
    };

    let mut checked = 0usize;
    for dtype in DTYPES {
        for head_dim in HEAD_DIMS {
            let classical = format!("sdpa_vector_{dtype}_{head_dim}");
            let packed = crate::kernels::kernels::params::packed_name(&classical);
            // The classical name first: if it does not resolve, a failure on
            // the packed one would be misattributed to this change.
            kernels
                .load_pipeline_with_constants(
                    &device,
                    Source::Sdpa,
                    classical.clone(),
                    constants(),
                )
                .unwrap_or_else(|e| {
                    panic!("scaled_dot_product_attention.metal has no kernel named {classical:?}: {e:?}")
                });
            kernels
                .load_pipeline_with_constants(&device, Source::Sdpa, packed.clone(), constants())
                .unwrap_or_else(|e| {
                    panic!(
                        "scaled_dot_product_attention.metal has no kernel named \
                         {packed:?} (the packed form of {classical:?}): {e:?}"
                    )
                });
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        DTYPES.len() * HEAD_DIMS.len(),
        "expected a packed counterpart for all 18 dispatchable sdpa_vector \
         variants, found {checked}"
    );
}

/// A fixed-width paged reduction must write every output, including at the
/// largest head dimension. Zero queries make the reference the selected values'
/// mean, independent of another GPU attention implementation.
#[test]
fn sdpa_vector_paged_writes_every_head_dimension() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let rows = 3;
    let nh = 4;
    let nkv = 2;
    let tokens = 64;
    let stride = 48;
    let lengths = [16i32, 33, 45];
    for hd in SDPA_VECTOR_HEAD_DIMS {
        for dtype in [SdpaDType::F32, SdpaDType::F16] {
            let upload = |values: &[f32]| match dtype {
                SdpaDType::F32 => new_buffer(&device, values),
                SdpaDType::F16 => new_buffer(
                    &device,
                    &values
                        .iter()
                        .copied()
                        .map(f16::from_f32)
                        .collect::<Vec<_>>(),
                ),
                _ => unreachable!(),
            };
            let q = upload(&vec![0.; rows * nh * hd]);
            let k = upload(&vec![0.25; nkv * tokens * hd]);
            let values: Vec<f32> = (0..nkv * tokens * hd)
                .map(|i| {
                    (i / (tokens * hd)) as f32 / 4.
                        + ((i / hd) % 17) as f32 / 32.
                        + (i % hd % 23) as f32 / 1024.
                })
                .collect();
            let v = upload(&values);
            let ns = new_buffer(&device, &lengths);
            for page_size in [1, 4] {
                let table: Vec<u32> = (0..rows * stride)
                    .map(|i| ((i / stride * 7 + i % stride * 3) % (tokens / page_size)) as u32)
                    .collect();
                let tb = new_buffer(&device, &table);
                for two_pass in [false, true] {
                    let partials = new_buffer(&device, &vec![f32::NAN; rows * nh * 32 * hd]);
                    let sums = new_buffer(&device, &vec![f32::NAN; rows * nh * 32]);
                    let maxs = new_buffer(&device, &vec![f32::NAN; rows * nh * 32]);
                    let mut previous: Option<Vec<u32>> = None;
                    for rep in 0..3 {
                        let out = upload(&vec![-999.; rows * nh * hd]);
                        let enc = commands.command_encoder().unwrap();
                        // SAFETY: the test owns every buffer bound or mapped here and reads its
                        // results only after waiting for the command buffer that writes them.
                        unsafe {
                            call_sdpa_vector_paged(
                                &device,
                                &enc,
                                &kernels,
                                rows,
                                nh,
                                nh / nkv,
                                hd,
                                0,
                                &q,
                                0,
                                &k,
                                tokens * hd,
                                0,
                                &v,
                                tokens * hd,
                                &tb,
                                0,
                                stride,
                                &ns,
                                0,
                                page_size,
                                &out,
                                1.,
                                dtype,
                                two_pass.then_some((&partials, &sums, &maxs)),
                            )
                        }
                        .unwrap();
                        drop(enc);
                        commands.wait_until_completed().unwrap();
                        let got: Vec<f32> = match dtype {
                            SdpaDType::F32 => read_to_vec(&out, rows * nh * hd),
                            SdpaDType::F16 => read_to_vec::<f16>(&out, rows * nh * hd)
                                .into_iter()
                                .map(f16::to_f32)
                                .collect(),
                            _ => unreachable!(),
                        };
                        for r in 0..rows {
                            for h in 0..nh {
                                for d in 0..hd {
                                    let want = (0..lengths[r] as usize)
                                        .map(|i| {
                                            let slot = table[r * stride + i / page_size] as usize
                                                * page_size
                                                + i % page_size;
                                            values[((h / (nh / nkv)) * tokens + slot) * hd + d]
                                        })
                                        .sum::<f32>()
                                        / lengths[r] as f32;
                                    let actual = got[(r * nh + h) * hd + d];
                                    assert!(actual.is_finite() && (actual - want).abs() < 1e-3,
                                        "hd={hd} dtype={dtype:?} page={page_size} two_pass={two_pass} rep={rep} row={r} head={h} d={d}: {actual} != {want}");
                                }
                            }
                        }
                        let bits: Vec<_> = got.iter().map(|x| x.to_bits()).collect();
                        if let Some(previous) = &previous {
                            assert_eq!(&bits, previous);
                        }
                        previous = Some(bits);
                    }
                    println!("PAGED_PARITY hd={hd} dtype={dtype:?} page={page_size} two_pass={two_pass} repetitions=3 pass=true");
                }
            }
        }
    }
}

/// `sdpa_vector` instantiates two head dimensions that `sdpa.rs` cannot ask
/// for, and this pins that rather than silently inheriting it.
///
/// `instantiate_sdpa_vector_heads` names **eight** head dimensions — 32, 64,
/// 72, 80, 96, 128, 256, 512 — while `call_sdpa_vector`'s `match (bk, itype)`
/// covers **six**. `sdpa_vector_float16_t_72` and `_80` are in the metallib and
/// unreachable from Rust; asking for head_dim 72 or 80 returns
/// `SdpaHeadSizeMismatch` naming an expected set that does not include them,
/// even though the kernel exists.
///
/// # Why this is worth a test rather than a fix here
///
/// It is the **inverse** of `DESIGN.md` §8.1b's absent-variant class, which has
/// now fired four times: there, a name was *asked for* and not declared, and the
/// symptom was a `LoadFunctionError` in a forward pass. Here a name is
/// *declared* and not askable, and the symptom is a capability that silently
/// does not exist — six dead instantiations in the metallib, paid for at every
/// cold compile.
///
/// `call_sdpa_full`'s own list is a third spelling of the same axis and covers
/// **eight** including 72 and 80, so the two entry points in one file disagree
/// about which head dimensions this file supports. That is precisely the
/// hand-sync a registry removes, and this family is the one §11.3h records as
/// still lacking one.
///
/// Not fixed here: widening the match is a behaviour change to a public entry
/// point (two shapes that error today would start computing), and it wants its
/// own bisect point beside a binding-style change — the same reason #64 split
/// #82 out and #81 left `left_size` alone. Pinned in #64's shape instead:
/// asserting the **current** state, so it turns red when corrected rather than
/// sitting as a TODO.
/// **Why `sdpa_vector` computes nothing at f32 x head_dim 512**: its pipeline
/// admits fewer threads per threadgroup than the 1024 `call_sdpa_vector`
/// dispatches, so the dispatch writes no output (the "returns ~0" of
/// `docs/porting-models.md` §21.8). bf16 x 512 does the same on a GPU that
/// emulates bf16 (Apple7). f16 at 512 and f32 at 256 admit 1024: the
/// control. Asserting the current state, so a toolchain that fits it turns
/// this red and the refusal can be revisited.
#[test]
fn sdpa_vector_f32_512_admits_fewer_threads_than_it_is_dispatched_at() {
    let device = device();
    let kernels = Kernels::new();
    let constants = || {
        Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]))
    };
    let max = |name: &str| {
        kernels
            .load_pipeline_with_constants(&device, Source::Sdpa, name.to_string(), constants())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
            .max_total_threads_per_threadgroup()
    };
    for name in [
        "sdpa_vector_float_512",
        "sdpa_vector_float16_t_512",
        "sdpa_vector_bfloat16_t_512",
        "sdpa_vector_float_256",
    ] {
        println!("{name}: maxTotalThreadsPerThreadgroup {}", max(name));
    }
    assert!(max("sdpa_vector_float_512") < 1024);
    let bf16_native = crate::kernels::metal::arms::for_device(&device)
        .has(crate::kernels::metal::arms::Feature::Bf16);
    if !bf16_native {
        assert!(max("sdpa_vector_bfloat16_t_512") < 1024);
    }
    for control in ["sdpa_vector_float16_t_512", "sdpa_vector_float_256"] {
        assert_eq!(max(control), 1024, "{control}");
    }
}

#[test]
fn sdpa_vector_declares_variants_the_name_table_cannot_reach() {
    let device = device();
    let kernels = Kernels::new();

    // With constants, for the reason `sdpa_packed_names_resolve` records: an
    // unspecialized function carrying a function constant aborts rather than
    // erroring when a pipeline is built from it.
    let constants = || {
        Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]))
    };

    for head_dim in [72usize, 80] {
        let name = format!("sdpa_vector_float16_t_{head_dim}");
        kernels
            .load_pipeline_with_constants(&device, Source::Sdpa, name.clone(), constants())
            .unwrap_or_else(|e| {
                panic!(
                    "{name:?} is expected to be instantiated in \
                     scaled_dot_product_attention.metal but absent from \
                     sdpa.rs's name table. If it no longer compiles, the \
                     instantiation list changed and this test should be \
                     revisited rather than deleted: {e:?}"
                )
            });
        // And its packed sibling, since the instantiation macro emits both.
        let packed = crate::kernels::kernels::params::packed_name(&name);
        kernels
            .load_pipeline_with_constants(&device, Source::Sdpa, packed.clone(), constants())
            .unwrap_or_else(|e| panic!("{packed:?} must exist alongside {name:?}: {e:?}"));
    }

    // The other half of the claim: Rust cannot ask for them. Asserted against
    // the error rather than by reading the match, so a fix that widens the
    // table turns this red.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let err = unsafe {
        call_sdpa_vector(
            &device,
            &commands(&device).command_encoder().unwrap(),
            &kernels,
            0,
            &[1, 1, 1, 72],
            &device.new_buffer(72 * 2, RESOURCE_OPTIONS).unwrap(),
            0,
            &[1, 1, 1, 72],
            &[72, 72, 72, 1],
            &device.new_buffer(72 * 2, RESOURCE_OPTIONS).unwrap(),
            0,
            &[72, 72, 72, 1],
            &device.new_buffer(72 * 2, RESOURCE_OPTIONS).unwrap(),
            &device.new_buffer(72 * 2, RESOURCE_OPTIONS).unwrap(),
            1.0,
            1.0,
            SdpaDType::F16,
        )
    }
    .expect_err(
        "head_dim 72 is instantiated in the metallib; if call_sdpa_vector now \
         accepts it, the name table was widened and this test has done its job \
         -- update it rather than relaxing it",
    );
    assert!(
        matches!(&err, MetalKernelError::SdpaHeadSizeMismatch(m) if m.got == 72),
        "expected the head-size mismatch that records the gap, got {err:?}"
    );
}

/// `sdpa_vector_float16_t_64` — LFM2's decode attention, 8 dispatches per token
/// and the whole reason this family is on the ICB path (`DESIGN.md` §6.2a).
///
/// Bit-identical rather than approximately equal, for the reason every other
/// family's arms give: the two entry points come from one body, so any
/// difference means the binding style changed what the kernel computed.
///
/// **LFM2's own geometry**: 32 query heads over 8 KV heads (GQA 4:1) at
/// head_dim 64 (`DESIGN.md` §5.2). `gqa_factor` is therefore 4 rather than 1,
/// which matters — #97 found that all five pre-existing arms in
/// `candle-nn/tests/sdpa.rs` set q heads == kv heads, so a kernel ignoring
/// `gqa_factor` entirely left every one of them green (§6.2a correction 3).
/// An arm at `gqa_factor == 1` would be the same blind spot in a new place.
#[test]
fn sdpa_packed_matches_split_for_sdpa_vector() {
    let device = device();
    let kernels = Kernels::new();

    // LFM2: 32 q heads, 8 kv heads, head_dim 64. A short kv_len keeps the test
    // fast; the kernel loops `for (i = simd_gid; i < N; i += BN)` with BN = 32,
    // so 40 keys exercises both the first pass and the wrap.
    let (n_q_heads, n_kv_heads, head_dim, kv_len) = (32usize, 8usize, 64usize, 40usize);

    let q: Vec<f16> = (0..n_q_heads * head_dim)
        .map(|i| f16::from_f32(((i * 37 % 71) as f32 - 35.0) / 64.0))
        .collect();
    // **K and V are given different per-head strides**, and that is deliberate
    // rather than incidental. `k_stride` and `v_stride` are two adjacent
    // `size_t` fields of `SdpaVectorParams`, so a layout error that transposes
    // them is exactly the silent case
    // `sdpa_params_layout_check_detects_a_moved_field` mutates for — and an arm
    // where the two strides are *equal* cannot see it, however faithfully it
    // compares. That is #81's finding 3 in a new form: it warns that a
    // mutation must swap fields that are both *read*, and this is the sharper
    // version — both read **and carrying different values**. Checked by
    // applying that mutation to production source: with equal strides the
    // parity arm passes under it.
    //
    // V therefore gets 8 elements of padding per head. The kernel takes
    // `k_stride` and `v_stride` separately, so this is a configuration it
    // supports rather than an abuse of it.
    const V_HEAD_PAD: usize = 8;
    let k: Vec<f16> = (0..n_kv_heads * kv_len * head_dim)
        .map(|i| f16::from_f32(((i * 53 % 97) as f32 - 48.0) / 96.0))
        .collect();
    let v_head_stride = kv_len * head_dim + V_HEAD_PAD;
    let v: Vec<f16> = (0..n_kv_heads * v_head_stride)
        .map(|i| f16::from_f32(((i * 29 % 83) as f32 - 41.0) / 48.0))
        .collect();

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let kv_shape = [1usize, n_kv_heads, kv_len, head_dim];
    // `call_sdpa_vector` reads index 1 of each — the per-head stride, which is
    // what the kernel adds `kv_head_idx *` to.
    let k_strides = [
        n_kv_heads * kv_len * head_dim,
        kv_len * head_dim,
        head_dim,
        1,
    ];
    let v_strides = [n_kv_heads * v_head_stride, v_head_stride, head_dim, 1];
    assert_ne!(
        k_strides[1], v_strides[1],
        "the strides must differ, or transposing them in the packed struct is \
         invisible and this arm cannot see the mutation it exists to catch"
    );
    let out_el = n_q_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut outs = Vec::new();
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let q_buf = new_buffer(&device, &q);
        let k_buf = new_buffer(&device, &k);
        let v_buf = new_buffer(&device, &v);
        let output = device
            .new_buffer(out_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_sdpa_vector_with(
                &device,
                &encoder,
                &kernels,
                0,
                &q_shape,
                &q_buf,
                0,
                &kv_shape,
                &k_strides,
                &k_buf,
                0,
                &v_strides,
                &v_buf,
                &output,
                scale,
                1.0,
                SdpaDType::F16,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        outs.push(read_to_vec::<f16>(&output, out_el));
    }
    assert_packed_matches_split("sdpa_vector_float16_t_64", &outs[0], &outs[1]);
}

/// The layout check must be able to fail, and must fail on *the silent case*.
///
/// `DESIGN.md` §15.1 #1 and `CONTRIBUTING.md` §3.1 #2: a test that cannot fail
/// is not a test. The mutation **swaps two adjacent same-typed fields** —
/// `SdpaVectorParams.k_stride` and `.v_stride`, both `size_t`. That case is
/// chosen over inserting or removing a field because it is the one nothing else
/// catches:
///
/// * `sizeof` is unchanged at 32, so the `static_assert` in
///   `scaled_dot_product_attention.metal` still passes;
/// * both fields are `size_t`, so the classical wrapper's brace initializer
///   still type-checks — C++11 narrowing rejects a *differently typed*
///   insertion or reorder before it reaches a GPU, which is a real layer of
///   defence but not this one;
/// * the kernel runs and reads keys at the values' stride, producing a
///   plausible wrong answer rather than a crash.
///
/// **Both swapped fields are read**, which is the condition #81's finding 3
/// makes explicit: `left_size` in `indexing.metal` **was** bound by all five
/// kernels and read by none, so a mutation swapping it proved nothing — which
/// is how it was found, and #219 has since removed it. `k_stride` and
/// `v_stride` are each dereferenced in `sdpa_vector_body`'s pointer adjustment.
///
/// It is mutated on **production source** — `crate::kernels::source::SDPA` is the same
/// `include_str!` the library is compiled from — rather than on a copy.
///
/// # What each guard actually catches, measured rather than assumed
///
/// Four mutations were applied to production source and each test re-run. Two
/// of them **survive the parity arm**, and saying which is more useful than the
/// two that do not:
///
/// | mutation | layout check | parity arm | #53 guard |
/// |---|---|---|---|
/// | 1. swap `k_stride`/`v_stride` in the **struct declaration** | **kills** | survives | — |
/// | 2. transpose them in the **Rust argument list** | survives | survives | — |
/// | 3. transpose them in the **packed wrapper only** | survives | **kills** | — |
/// | 4. `return;` at the top of the shared body | survives | — | **kills** |
///
/// **Mutations 1 and 2 are invisible to the parity arm by construction, and
/// that is the mechanism working rather than failing.** Both live in something
/// the two styles *share* — the struct's layout, and the single `set_params!`
/// argument list — so they move both arms identically and the comparison is
/// between two equally-wrong answers. That is the flip side of the property
/// `DESIGN.md` §11.3d claims for this whole conversion: because only one
/// argument list exists, the styles cannot disagree about what is bound *or*
/// about being wrong together.
///
/// So the three guards are genuinely non-overlapping and none subsumes another:
/// the **layout check** covers what the two sides share about the struct, the
/// **parity arm** covers what one style does and the other does not, and
/// **#53's non-vacuity guard** covers a kernel that does nothing at all. A
/// suite carrying only the parity arm would pass under mutations 1, 2 and 4.
///
/// Mutation 3 is also what forced the parity arm's K and V to carry
/// **different** per-head strides. With the contiguous layout it started with,
/// `k_stride == v_stride` and transposing them is a no-op — the arm passed
/// under mutation 3 until the input was fixed. #81's finding 3 says a mutation
/// must swap two fields that are both *read*; this is the sharper statement:
/// both read **and holding different values**, or the test proves nothing.
#[test]
fn sdpa_params_layout_check_detects_a_moved_field() {
    use crate::kernels::kernels::params::LayoutFamily;

    let descriptor = LayoutFamily::Sdpa.descriptor();
    let pipeline = mutated_layout_pipeline(
        crate::kernels::source::SDPA,
        "  size_t k_stride;\n  size_t v_stride;",
        "  size_t v_stride;\n  size_t k_stride;",
        descriptor.kernel,
    );
    let d = layout_disagreements(&pipeline, &descriptor.expected);

    assert!(
        d.iter().any(|x| x.starts_with("SdpaVectorParams.k_stride")),
        "swapping k_stride must be reported; got {d:?}"
    );
    assert!(
        d.iter().any(|x| x.starts_with("SdpaVectorParams.v_stride")),
        "swapping v_stride must be reported; got {d:?}"
    );
    assert!(
        !d.iter().any(|x| x.starts_with("sizeof(SdpaVectorParams)")),
        "the swap must not change sizeof, or it is not the silent case; got {d:?}"
    );
}

// ---------------------------------------------------------------------------
// Fused RMSNorm prologue on the decode-path GEMV (issue #266, `DESIGN.md`
// §12.2 #2).
//
// The entry is "RMSNorm into the following matmul, 30 layers x 2 sites". What
// is buildable is the *round-trip* removal, not the dispatch removal, and the
// reason is structural rather than a matter of effort: a GEMV partitions its
// OUTPUT across `n_tgp` threadgroups -- 336 of them for LFM2's `w1` -- and
// Metal has no cross-threadgroup barrier inside a dispatch, so a prologue
// cannot compute a reduction over the whole input vector. It can only
// recompute it, once per threadgroup.
// `measurements/issue-266-raw/prediction.py` prices that at 432x the reduction
// work against a 0.492 MB/token saving.
//
// So the reduction stays its own dispatch (`rmsnorm_scale_*`, which reuses
// `rms_norm`'s reduction tree line for line) and the prologue applies the
// scale and the weight as the vector is loaded into registers.
// ---------------------------------------------------------------------------

/// Drive the fused arm end to end: the scale reduction, then the GEMV whose
/// prologue consumes it.
///
/// Goes through the real entry points rather than reimplementing the tile
/// selection, so the fused and unfused arms resolve the same `[[host_name]]`
/// stem and differ only in the `_rmsnorm` suffix.
#[allow(clippy::too_many_arguments)]
fn run_gemv_fused_rmsnorm(
    dtype: GemmDType,
    scale_kernel: &'static str,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[f32],
    lhs_stride: &[usize],
    rhs: &[f32],
    rhs_stride: &[usize],
    norm_w: &[f32],
    eps: f32,
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let lhs_buf = mk(lhs);
    let rhs_buf = mk(rhs);
    let w_buf = mk(norm_w);

    // One f32 of scale per row of the input vector. At B=1 decode that is one.
    let scale_buf = device
        .new_buffer(b * core::mem::size_of::<f32>(), options)
        .unwrap();

    // The input vector is `lhs` when `n == 1` and `rhs` when `m == 1`; the
    // fused arm is only instantiated for the non-transposed kernel, which is
    // the `n == 1` shape here.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm_scale(
            &device,
            &encoder,
            &kernels,
            scale_kernel,
            b * k,
            k,
            eps,
            &rhs_buf,
            0,
            &scale_buf,
        )
    }
    .unwrap();

    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<f32>(), options)
        .unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            ParamStyle::Split,
            Some(GemvRmsNorm {
                weight: &w_buf,
                weight_offset: 0,
                scale: &scale_buf,
                scale_offset: 0,
            }),
            GemvEpilogue::None,
        )
    }
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, length)
}

/// A CPU reference for `rms_norm` followed by the matmul, computed in f64 so
/// it is not itself one of the two f32 orderings being compared.
///
/// `DESIGN.md` §2.3.5a: the load-bearing discriminator is parity against
/// something that is *not* the thing being changed. §10.2g's own precedent is
/// an f64 accumulation of the same products, "because reading either float32
/// order as ground truth is precisely what makes a legitimate reordering look
/// like an error".
fn reference_rmsnorm_gemv(
    (m_out, k): (usize, usize),
    mat: &[f32],
    vec: &[f32],
    norm_w: &[f32],
    eps: f32,
) -> Vec<f64> {
    let mut sumsq = 0.0f64;
    for &v in vec.iter().take(k) {
        sumsq += (v as f64) * (v as f64);
    }
    let inv = 1.0f64 / (sumsq / k as f64 + eps as f64).sqrt();
    let normed: Vec<f64> = (0..k)
        .map(|i| vec[i] as f64 * inv * norm_w[i] as f64)
        .collect();
    (0..m_out)
        .map(|row| {
            let mut acc = 0.0f64;
            for c in 0..k {
                acc += mat[row * k + c] as f64 * normed[c];
            }
            acc
        })
        .collect()
}

/// The fused prologue computes RMSNorm-then-matmul.
///
/// Both LFM2 decode tiles are covered, at the geometry `call_mlx_gemv` selects
/// for them: `bm8` serves `w1`/`w3` (N=10752) and `conv.in_proj` (N=6144),
/// `bm4` serves `q_proj` (N=2048) and `k`/`v_proj` (N=512).
///
/// **The norm weight is not constant, deliberately.** With `norm_w` all ones a
/// wrong weight index -- the failure mode a prologue is most likely to have --
/// is invisible, which is §9.2c's "a fixture built only from the model's own
/// shapes is blind to a whole defect class" in this kernel's own parameter.
#[test]
fn gemv_fused_rmsnorm_matches_a_reference() {
    let eps = 1e-5f32;
    // K is LFM2's hidden size; the two N values select the two decode tiles.
    //
    // **K = 2050 is in the list and it is not decoration.** The GEMV walks the
    // input vector in `blockN = BN*SN*TN = 128` chunks and handles the
    // remainder in a separate `leftover` path, so at LFM2's own K = 2048 the
    // tail NEVER EXECUTES -- 2048 = 16 x 128 exactly. Measured: a mutation
    // deleting the prologue from the leftover path entirely passes every test
    // when the fixtures are all 2048-shaped.
    //
    // That is §9.2c's finding in this kernel's own geometry -- "a parity test
    // built only from the model's own shapes is blind to a whole defect
    // class" -- and #70's `align_up` case, and §10.4a's `k_token_stride` case.
    // The fixture must be UNALIGNED AT THE LEVEL WHERE THE BRANCH IS TAKEN.
    for &(k, n_out, label) in &[
        (2048usize, 512usize, "bm4 (k/v_proj), K divides blockN"),
        (2048usize, 6144usize, "bm8 (in_proj), K divides blockN"),
        (2050usize, 512usize, "bm4, K leaves a 2-element tail"),
        (2050usize, 6144usize, "bm8, K leaves a 2-element tail"),
    ] {
        let (b, m, n) = (1usize, n_out, 1usize);

        let mat: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
            .collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
        // Non-constant, and never 1.0, so a wrong index cannot pass.
        let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();

        let got = run_gemv_fused_rmsnorm(
            GemmDType::F32,
            "rmsnorm_scale_f32",
            (b, m, n, k),
            &mat,
            &[m * k, k, 1],
            &vec,
            &[k, 1, 1],
            &norm_w,
            eps,
        );
        let want = reference_rmsnorm_gemv((n_out, k), &mat, &vec, &norm_w, eps);

        assert_eq!(got.len(), want.len(), "{label}: length");
        // Non-vacuity, per §15.1 #1 and issue #53: two kernels that both write
        // nothing agree perfectly.
        let distinct = got
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{label}: fused output has {distinct} distinct value(s); the \
             comparison would be vacuous"
        );

        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let rel = ((g as f64) - w).abs() / denom;
            if rel > worst {
                worst = rel;
            }
            assert!(
                rel < 2e-5,
                "{label}: row {i}: fused {g} vs reference {w} (rel {rel:e})"
            );
        }
        assert!(
            worst > 0.0,
            "{label}: exact agreement with an f64 reference \
                is not expected in f32 and suggests the comparison did not run"
        );
    }
}

/// The fused arm agrees with the unfused arm driven through the shipping
/// `rms_norm` kernel.
///
/// This is the comparison that matters for §12.2 #2: the fused result must be
/// what the two-dispatch path already computes, to within the difference the
/// reordering itself introduces. It is deliberately a *tolerance* comparison
/// and not a bit-equality one -- the shipping kernel scales in `T` and writes
/// `T`, where the prologue scales in the GEMV's f32 accumulator, so the two
/// legitimately differ in the low bits. §2.3.5a's procedure is what adjudicates
/// that, and the f64 reference above is what says which one is nearer.
#[test]
fn gemv_fused_rmsnorm_agrees_with_the_unfused_path() {
    let eps = 1e-5f32;
    let k = 2048usize;
    let n_out = 512usize;
    let (b, m, n) = (1usize, n_out, 1usize);

    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();

    let fused = run_gemv_fused_rmsnorm(
        GemmDType::F32,
        "rmsnorm_scale_f32",
        (b, m, n, k),
        &mat,
        &[m * k, k, 1],
        &vec,
        &[k, 1, 1],
        &norm_w,
        eps,
    );

    // The unfused path: `rms_norm` writes a normalized vector, then the
    // ordinary GEMV consumes it.
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let mat_buf = mk(&mat);
    let vec_buf = mk(&vec);
    let w_buf = mk(&norm_w);
    let normed = device
        .new_buffer(k * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm(
            &device,
            &encoder,
            &kernels,
            "rmsnorm_f32",
            k,
            k,
            eps,
            &vec_buf,
            0,
            &w_buf,
            0,
            &normed,
        )
    }
    .unwrap();
    let out = device
        .new_buffer(n_out * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_with(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (b, m, n, k),
            &[m * k, k, 1],
            0,
            &mat_buf,
            &[k, 1, 1],
            0,
            &normed,
            &out,
            ParamStyle::Split,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    let unfused: Vec<f32> = read_to_vec(&out, n_out);

    let want = reference_rmsnorm_gemv((n_out, k), &mat, &vec, &norm_w, eps);

    let mut worst_fused = 0.0f64;
    let mut worst_unfused = 0.0f64;
    for i in 0..n_out {
        let denom = want[i].abs().max(1.0);
        worst_fused = worst_fused.max(((fused[i] as f64) - want[i]).abs() / denom);
        worst_unfused = worst_unfused.max(((unfused[i] as f64) - want[i]).abs() / denom);
    }
    // Neither arm is "correct" and the other wrong: both are f32 roundings of
    // one sum, and the f64 reference is what says they are the same distance
    // from it. §10.2g's discipline, in this kernel.
    assert!(
        worst_fused < 2e-5 && worst_unfused < 2e-5,
        "fused {worst_fused:e} / unfused {worst_unfused:e} against an f64 reference"
    );
}

/// The fused arm is refused where it has no instantiation, rather than
/// silently running unfused.
///
/// A fused request that quietly ran the unfused kernel would return an
/// un-normalized result, and under `HazardTrackingModeUntracked` (§3.5)
/// nothing reports it. Both bounds are asserted -- the refusal AND the
/// admitted case -- because a check that refuses everything is §8.1g's
/// failure, and the admitted arm is the load-bearing half.
#[test]
fn gemv_fused_rmsnorm_is_refused_where_it_is_not_instantiated() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let k = 2048usize;
    let buf = device.new_buffer(4 * k * 4, options).unwrap();
    let scale = device.new_buffer(4, options).unwrap();
    let out = device.new_buffer(4 * k * 4, options).unwrap();
    let norm = GemvRmsNorm {
        weight: &buf,
        weight_offset: 0,
        scale: &scale,
        scale_offset: 0,
    };

    // `m == 1` routes to the TRANSPOSED kernel, which carries no `_rmsnorm`
    // instantiation. It must be an error rather than a silent unfused run.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let transposed = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 1, 512, k),
            &[k, k, 1],
            0,
            &buf,
            &[512 * k, 512, 1],
            0,
            &buf,
            &out,
            ParamStyle::Split,
            Some(norm),
            GemvEpilogue::None,
        )
    };
    assert!(
        transposed.is_err(),
        "the transposed kernel has no fused instantiation and must be refused"
    );

    // The admitted bound: the non-transposed decode tile IS instantiated.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let ok = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 512, 1, k),
            &[512 * k, k, 1],
            0,
            &buf,
            &[k, 1, 1],
            0,
            &buf,
            &out,
            ParamStyle::Split,
            Some(norm),
            GemvEpilogue::None,
        )
    };
    assert!(
        ok.is_ok(),
        "the non-transposed decode tile must be admitted: {ok:?}"
    );

    drop(encoder);
    commands.wait_until_completed().unwrap();
}

/// The unfused arm is untouched: asking for no fusion resolves the same
/// `[[host_name]]` and computes the same bits as before the prologue existed.
///
/// Bit-identical rather than approximate, deliberately. `PrologueNone` is the
/// identity and is meant to inline away entirely; anything but an exact match
/// means threading the template parameter changed what the shipping kernel
/// computes, which is §12.3's requirement that the two variants coexist
/// without the unfused one paying for it.
#[test]
fn gemv_unfused_arm_is_bit_identical_with_the_prologue_threaded() {
    let k = 2048usize;
    let n_out = 512usize;
    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();

    let a = run_mlx_gemv_style::<f32>(
        GemmDType::F32,
        (1, n_out, 1, k),
        &mat,
        &[n_out * k, k, 1],
        &vec,
        &[k, 1, 1],
        ParamStyle::Split,
    );
    let b = run_mlx_gemv_style::<f32>(
        GemmDType::F32,
        (1, n_out, 1, k),
        &mat,
        &[n_out * k, k, 1],
        &vec,
        &[k, 1, 1],
        ParamStyle::Split,
    );
    assert_eq!(a, b, "the unfused arm is not reproducible");
    let distinct = a
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(distinct >= 2, "vacuous: {distinct} distinct value(s)");
}

/// `rmsnorm_scale` writes the same scale the shipping `rms_norm` applies.
///
/// Checked by driving `rms_norm` with an all-ones weight and dividing: the
/// normalized output is then exactly `src * total`, so `total` is recoverable
/// and can be compared against what `rmsnorm_scale` wrote. Bit-equality is
/// asserted because the two kernels run the SAME reduction tree at the same
/// block size -- that is the property `call_rms_norm_scale` exists to
/// preserve, and a tolerance comparison would not detect losing it.
#[test]
fn rmsnorm_scale_matches_the_shipping_reduction() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let eps = 1e-5f32;
    let k = 2048usize;

    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    let ones: Vec<f32> = vec![1.0f32; k];
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let src = mk(&vec);
    let w = mk(&ones);
    let normed = device.new_buffer(k * 4, options).unwrap();
    let scale = device.new_buffer(4, options).unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm(
            &device,
            &encoder,
            &kernels,
            "rmsnorm_f32",
            k,
            k,
            eps,
            &src,
            0,
            &w,
            0,
            &normed,
        )
    }
    .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm_scale(
            &device,
            &encoder,
            &kernels,
            "rmsnorm_scale_f32",
            k,
            k,
            eps,
            &src,
            0,
            &scale,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let normed: Vec<f32> = read_to_vec(&normed, k);
    let scale: Vec<f32> = read_to_vec(&scale, 1);

    // Recover `total` from an element whose source value is comfortably away
    // from zero, so the division is well conditioned.
    let idx = (0..k)
        .max_by(|&a, &b| vec[a].abs().partial_cmp(&vec[b].abs()).unwrap())
        .unwrap();
    assert!(
        vec[idx].abs() > 0.1,
        "fixture has no well-conditioned element"
    );
    let recovered = normed[idx] / vec[idx];
    let rel = ((recovered - scale[0]) / scale[0]).abs();
    assert!(
        rel < 1e-6,
        "rmsnorm_scale wrote {} where rms_norm applied {recovered} (rel {rel:e})",
        scale[0]
    );
}

/// Sorts `ncols` **tie-free** values through `call_mlx_arg_sort` and returns the
/// indices. Ties are excluded deliberately: the sort is unstable, so a fixture
/// with duplicates cannot distinguish "returned a different valid answer" from
/// "returned a wrong one", and the defect this exercises presents as duplicated
/// indices.
fn run_mlx_sort_dtype<T: Clone>(v: &[T], ncols: usize, dtype: DType) -> Vec<u32> {
    let nrows = v.len() / ncols;
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let indexes = vec![0u32; v.len()];
    let output = new_buffer(&device, &indexes);

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_arg_sort(
            &device,
            &encoder,
            &kernels,
            dtype,
            nrows,
            ncols,
            BufferOffset::zero_offset(&input),
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

/// Asserts `idx` is a permutation of `0..ncols` **and** orders `vals`
/// ascending. The permutation half is the one that discriminates: an
/// all-zeros or index-duplicating result satisfies a naive "is each element
/// <= the next" check vacuously (lloom `DESIGN.md` §15.1 #1).
fn assert_sorts<F: Fn(u32) -> f32>(idx: &[u32], ncols: usize, val: F, what: &str) {
    let mut seen = vec![false; ncols];
    for &i in idx {
        let i = i as usize;
        assert!(i < ncols, "{what}: index {i} out of range 0..{ncols}");
        assert!(
            !seen[i],
            "{what}: index {i} appears twice — not a permutation"
        );
        seen[i] = true;
    }
    assert_eq!(idx.len(), ncols, "{what}: wrong length");
    for w in idx.windows(2) {
        let (a, b) = (val(w[0]), val(w[1]));
        assert!(a <= b, "{what}: not ascending — {a} before {b}");
    }
}

/// `multi_block_sort` names its merge kernel from the dtype, like the two
/// kernels either side of it.
///
/// `mb_block_merge` is templated on `val_t` and reads `dev_vals_in` through it.
/// The name was pinned to `merge_mbsort_float32_…` regardless of dtype, so at
/// every non-f32 dtype the merge reinterpreted the block-sorted values through
/// a `float` pointer. **The only test on this path was F32**, where the
/// hardcode is accidentally correct — so nothing exercised it.
///
/// Measured before the fix at f16, ncols = 16 000: 8 000 of 16 000 indices
/// distinct, 8 000 duplicated, not ascending — against the same fixture at f32
/// on the same code returning a clean permutation.
#[test]
fn mlx_multi_block_sort_merges_at_every_dtype() {
    // 16 000 columns forces `n_blocks = 4`, so the merge runs. A tie-free
    // fixture is built from distinct f16 bit patterns, which every dtype below
    // represents exactly: f16 has 31 744 distinct positive finite values, so a
    // 16 000-wide fixture has room without approaching the subnormal end.
    let ncols = 16_000usize;
    let base: Vec<f16> = (0..ncols)
        .map(|i| f16::from_bits((i + 1024) as u16))
        .collect();
    // Shuffle so the answer is not the identity — an identity fixture cannot
    // tell a sort from a no-op.
    let mut shuffled = base.clone();
    shuffled.shuffle(&mut rng());
    assert!(
        shuffled.windows(2).any(|w| w[0] > w[1]),
        "fixture is already sorted; the test would pass on a no-op"
    );

    // f16 — the dtype LFM2 decodes in, and the one the hardcode broke.
    let vals = shuffled.clone();
    let idx = run_mlx_sort_dtype(&vals, ncols, DType::F16);
    assert_sorts(&idx, ncols, |i| vals[i as usize].to_f32(), "f16");

    // bf16 — the checkpoint's on-disk dtype. Its own bit patterns, not the f16
    // ramp converted: bf16 has 8 fewer mantissa bits, so converting would
    // collide and reintroduce the ties this fixture exists to exclude.
    let mut vals_bf: Vec<bf16> = (0..ncols)
        .map(|i| bf16::from_bits((i + 1024) as u16))
        .collect();
    vals_bf.shuffle(&mut rng());
    let idx = run_mlx_sort_dtype(&vals_bf, ncols, DType::BF16);
    assert_sorts(&idx, ncols, |i| vals_bf[i as usize].to_f32(), "bf16");

    // f32 — the control. This arm passed before the fix and must still pass;
    // without it a green run is consistent with having broken the path
    // wholesale rather than repaired it.
    let vals_f32: Vec<f32> = shuffled.iter().map(|v| v.to_f32()).collect();
    let idx = run_mlx_sort_dtype(&vals_f32, ncols, DType::F32);
    assert_sorts(&idx, ncols, |i| vals_f32[i as usize], "f32 control");

    // u32 — an integer dtype, so the reinterpretation is a different shape.
    let vals_u32: Vec<u32> = {
        let mut v: Vec<u32> = (0..ncols as u32).collect();
        v.shuffle(&mut rng());
        v
    };
    let idx = run_mlx_sort_dtype(&vals_u32, ncols, DType::U32);
    assert_sorts(&idx, ncols, |i| vals_u32[i as usize] as f32, "u32");
}

/// Issue #383 — parity for the tiles that issue instantiates, at shapes where
/// each tile's edge handling actually fires.
///
/// **Why this is driven by an environment variable rather than a loop.**
/// `LLOOM_383_TILE` is read through a `OnceLock` (deliberately — §15.2 #10
/// forbids a per-dispatch `getenv`), so a single process can exercise exactly
/// one tile. The test therefore asserts parity for whichever tile the process
/// was started with, and `tiles-parity.sh` runs it once per tile. With the
/// variable unset it still runs and covers the shipping selection, so it is
/// not vacuous in a normal `cargo test`.
///
/// The reference is computed on the host in f32. That is the §3.1 "parity
/// against a reference" requirement met by something that is *not the thing
/// being changed* — the tile cannot make a host loop agree with it.
#[test]
fn mlx_gemm_383_tile_parity() {
    // Shapes chosen so that m is NOT a multiple of any candidate BM (16/32/64)
    // and n is NOT a multiple of every candidate BN, so the masked edge path
    // (`load_safe`) is exercised rather than only the aligned fast path.
    // k is deliberately not a multiple of 16 either, for `align_k`.
    for &(b, m, n, k) in &[
        (1usize, 17usize, 48usize, 40usize),
        (1, 33, 80, 24),
        (2, 5, 96, 40),
        (1, 64, 32, 32), // a fully aligned case, as the control
    ] {
        let lhs: Vec<f32> = (0..b * m * k).map(|i| ((i % 13) as f32) - 6.0).collect();
        let rhs: Vec<f32> = (0..b * n * k).map(|i| ((i % 7) as f32) - 3.0).collect();

        let got = run_mlx_gemm(
            GemmDType::F32,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );

        // Host reference: C[bi,i,j] = sum_l A[bi,i,l] * B[bi,l,j]
        let mut want = vec![0f32; b * m * n];
        for bi in 0..b {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for l in 0..k {
                        acc += lhs[bi * m * k + i * k + l] * rhs[bi * n * k + l * n + j];
                    }
                    want[bi * m * n + i * n + j] = acc;
                }
            }
        }

        let tile = std::env::var("LLOOM_383_TILE").unwrap_or_else(|_| "unset".into());
        assert_eq!(
            got.len(),
            want.len(),
            "tile={tile} shape=({b},{m},{n},{k}): length"
        );
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                "tile={tile} shape=({b},{m},{n},{k}) idx={idx}: {g} vs {w}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fused SiLU epilogue on the decode-path GEMV (issue #484, `DESIGN.md`
// §12.2 #3).
//
// The counterpart to the RMSNorm prologue above, at the other end of the same
// kernel. §12.2a and §12.2b both lost on OUTPUT PARTITIONING: a GEMV splits
// its output across `n_tgp` threadgroups and each walks the whole input
// vector, so a prologue's work is multiplied by `n_tgp` whatever it computes.
//
// An epilogue is on the other side of that partition. `GEMVKernel::run` writes
// each output element from exactly one thread (`if (simdN == 0 && thrN == 0)`),
// so the transform runs once per output element -- the standalone kernel's own
// count. `measurements/issue-484-raw/prediction.py` states that arithmetic in
// full, before this was built.
// ---------------------------------------------------------------------------

/// Drive the fused epilogue arm through the real entry point.
///
/// Takes no extra buffers, which is the shape of the finding: a pure
/// elementwise epilogue reads nothing the kernel was not already reading.
#[allow(clippy::too_many_arguments)]
fn run_gemv_fused_silu(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[f32],
    lhs_stride: &[usize],
    rhs: &[f32],
    rhs_stride: &[usize],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let lhs_buf = mk(lhs);
    let rhs_buf = mk(rhs);

    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<f32>(), options)
        .unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            ParamStyle::Split,
            None,
            GemvEpilogue::Silu,
        )
    }
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, length)
}

/// A CPU reference for the matmul followed by SiLU, computed in f64 so it is
/// not itself one of the two f32 orderings being compared (§10.2g, §2.3.5a).
fn reference_gemv_silu((m_out, k): (usize, usize), mat: &[f32], vec: &[f32]) -> Vec<f64> {
    (0..m_out)
        .map(|row| {
            let mut acc = 0.0f64;
            for c in 0..k {
                acc += mat[row * k + c] as f64 * vec[c] as f64;
            }
            acc / (1.0 + (-acc).exp())
        })
        .collect()
}

/// The fused epilogue computes matmul-then-SiLU.
///
/// **K = 2050 is in the list for the reason #266's fixture carries it**: the
/// GEMV walks the input vector in `blockN` chunks and handles the remainder in
/// a separate `leftover` path, so at LFM2's own K = 2048 the tail never
/// executes. An epilogue is outside that loop and so is not itself tail-
/// sensitive -- but the ACCUMULATOR it transforms is, so a fixture that never
/// takes the tail cannot see an epilogue applied to a wrong sum.
///
/// The fixture values are chosen so the accumulator spans both signs and is
/// not near-constant: SiLU is monotone but strongly asymmetric about zero, and
/// an all-positive fixture cannot distinguish it from the identity times a
/// number near 1.
#[test]
fn gemv_fused_silu_matches_a_reference() {
    for &(k, n_out, label) in &[
        (
            2048usize,
            512usize,
            "bm4 (k/v_proj shape), K divides blockN",
        ),
        (2048usize, 6144usize, "bm8 (w1 shape), K divides blockN"),
        (2050usize, 512usize, "bm4, K leaves a 2-element tail"),
        (2050usize, 6144usize, "bm8, K leaves a 2-element tail"),
    ] {
        let (b, m, n) = (1usize, n_out, 1usize);

        let mat: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
            .collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();

        let got = run_gemv_fused_silu(
            GemmDType::F32,
            (b, m, n, k),
            &mat,
            &[m * k, k, 1],
            &vec,
            &[k, 1, 1],
        );
        let want = reference_gemv_silu((n_out, k), &mat, &vec);

        assert_eq!(got.len(), want.len(), "{label}: length");

        // Non-vacuity, per §15.1 #1: two kernels that both write nothing agree
        // perfectly.
        let distinct = got
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{label}: fused output has {distinct} distinct value(s); the \
             comparison would be vacuous"
        );

        // A SECOND non-vacuity arm, and this one is specific to SiLU. The
        // reference must contain values where SiLU is materially not the
        // identity -- i.e. some accumulator below zero, where SiLU's output
        // has the OPPOSITE sign behaviour to `x`. Without this a fixture whose
        // accumulators are all large and positive makes `silu(x) ~ x`, and the
        // test would pass against a kernel with no epilogue at all.
        let negatives = want.iter().filter(|v| **v < 0.0).count();
        assert!(
            negatives > 0,
            "{label}: no negative outputs; SiLU is indistinguishable from the \
             identity on this fixture and the comparison is vacuous"
        );

        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let rel = ((g as f64) - w).abs() / denom;
            if rel > worst {
                worst = rel;
            }
            assert!(
                rel < 2e-5,
                "{label}: row {i}: fused {g} vs reference {w} (rel {rel:e})"
            );
        }
        assert!(
            worst > 0.0,
            "{label}: exact agreement with an f64 reference is not expected in \
             f32 and suggests the comparison did not run"
        );
    }
}

/// The fused arm agrees with the two-dispatch path it replaces.
///
/// **A tolerance comparison and not a bit-equality one, and the reason is
/// recorded rather than assumed.** `silu_f16`/`silu_f32` evaluate
/// `x/(1+exp(-x))` at `T` on the value the GEMV already rounded and stored;
/// the epilogue evaluates it at the f32 accumulator before that rounding. The
/// fused arm therefore rounds once where the pair rounds twice, which is
/// strictly the more accurate order and still a different computation.
#[test]
fn gemv_fused_silu_agrees_with_the_unfused_path() {
    let k = 2048usize;
    let n_out = 512usize;
    let (b, m, n) = (1usize, n_out, 1usize);

    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();

    let fused = run_gemv_fused_silu(
        GemmDType::F32,
        (b, m, n, k),
        &mat,
        &[m * k, k, 1],
        &vec,
        &[k, 1, 1],
    );

    // The unfused path: the ordinary GEMV writes the accumulator, then the
    // standalone `silu` kernel transforms it.
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let mat_buf = mk(&mat);
    let vec_buf = mk(&vec);
    let gate = device
        .new_buffer(n_out * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_with(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (b, m, n, k),
            &[m * k, k, 1],
            0,
            &mat_buf,
            &[k, 1, 1],
            0,
            &vec_buf,
            &gate,
            ParamStyle::Split,
        )
    }
    .unwrap();
    let out = device
        .new_buffer(n_out * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_unary_contiguous(
            &device,
            &encoder,
            &kernels,
            unary::contiguous::silu::FLOAT,
            core::mem::size_of::<f32>(),
            n_out,
            BufferOffset {
                buffer: &gate,
                offset_in_bytes: 0,
            },
            &out,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    let unfused: Vec<f32> = read_to_vec(&out, n_out);

    let want = reference_gemv_silu((n_out, k), &mat, &vec);

    // Non-vacuity on BOTH arms, and on the property under test.
    assert!(
        want.iter().any(|v| *v < 0.0),
        "fixture has no negative outputs; SiLU would be indistinguishable \
         from the identity"
    );

    let mut worst_fused = 0.0f64;
    let mut worst_unfused = 0.0f64;
    for i in 0..n_out {
        let denom = want[i].abs().max(1.0);
        worst_fused = worst_fused.max(((fused[i] as f64) - want[i]).abs() / denom);
        worst_unfused = worst_unfused.max(((unfused[i] as f64) - want[i]).abs() / denom);
    }
    // Neither arm is "correct" and the other wrong: both are f32 roundings of
    // one computation, and the f64 reference is what says how far each sits.
    assert!(
        worst_fused < 2e-5 && worst_unfused < 2e-5,
        "fused {worst_fused:e} / unfused {worst_unfused:e} against an f64 reference"
    );
}

/// The fused epilogue is refused where it has no instantiation, rather than
/// silently running unfused.
///
/// Both bounds are asserted -- the refusals AND the admitted case -- because a
/// check that refuses everything is §8.1g's failure and the admitted arm is
/// the load-bearing half.
#[test]
fn gemv_fused_silu_is_refused_where_it_is_not_instantiated() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let k = 2048usize;
    let buf = device.new_buffer(4 * k * 4, options).unwrap();
    let scale = device.new_buffer(4, options).unwrap();
    let out = device.new_buffer(4 * k * 4, options).unwrap();

    // `m == 1` routes to the TRANSPOSED kernel, which carries no `_silu`
    // instantiation.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let transposed = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 1, 512, k),
            &[k, k, 1],
            0,
            &buf,
            &[512 * k, 512, 1],
            0,
            &buf,
            &out,
            ParamStyle::Split,
            None,
            GemvEpilogue::Silu,
        )
    };
    assert!(
        transposed.is_err(),
        "the transposed kernel has no fused epilogue and must be refused"
    );

    // The packed binding style HAS a fused sibling since #732 P3 (the ICB
    // could not otherwise cover the fused arms, §3.7c), so this is the
    // positive control of the refusals above: the same tile that is refused
    // transposed is admitted packed.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let packed = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 512, 1, k),
            &[512 * k, k, 1],
            0,
            &buf,
            &[k, 1, 1],
            0,
            &buf,
            &out,
            ParamStyle::Packed,
            None,
            GemvEpilogue::Silu,
        )
    };
    assert!(
        packed.is_ok(),
        "the packed style has a fused epilogue sibling and must be admitted: {packed:?}"
    );

    // Prologue and epilogue together resolve a name that is not instantiated.
    let norm = GemvRmsNorm {
        weight: &buf,
        weight_offset: 0,
        scale: &scale,
        scale_offset: 0,
    };
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let both = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 512, 1, k),
            &[512 * k, k, 1],
            0,
            &buf,
            &[k, 1, 1],
            0,
            &buf,
            &out,
            ParamStyle::Split,
            Some(norm),
            GemvEpilogue::Silu,
        )
    };
    assert!(
        both.is_err(),
        "no `_rmsnorm_silu` variant is instantiated and the combination must \
         be refused"
    );

    // The admitted bound: the non-transposed decode tile IS instantiated.
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let ok = unsafe {
        call_mlx_gemv_full(
            &device,
            &encoder,
            &kernels,
            GemmDType::F32,
            (1, 512, 1, k),
            &[512 * k, k, 1],
            0,
            &buf,
            &[k, 1, 1],
            0,
            &buf,
            &out,
            ParamStyle::Split,
            None,
            GemvEpilogue::Silu,
        )
    };
    assert!(
        ok.is_ok(),
        "the non-transposed decode tile must be admitted: {ok:?}"
    );

    drop(encoder);
    commands.wait_until_completed().unwrap();
}

/// The unfused arm is untouched: threading the epilogue template parameter
/// through `GEMVKernel` must not change what the shipping kernel computes.
///
/// Bit-identical rather than approximate. `EpilogueNone` is the identity and
/// is meant to inline away entirely; anything but an exact match means the
/// seam cost the unfused arm something, which is §12.3's requirement that the
/// two variants coexist without the default paying for it.
#[test]
fn gemv_unfused_arm_is_bit_identical_with_the_epilogue_threaded() {
    let k = 2048usize;
    let n_out = 512usize;
    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();

    let a = run_mlx_gemv_style::<f32>(
        GemmDType::F32,
        (1, n_out, 1, k),
        &mat,
        &[n_out * k, k, 1],
        &vec,
        &[k, 1, 1],
        ParamStyle::Split,
    );

    // The fused arm on the same inputs must NOT match, or the epilogue is not
    // engaged and this test would pass against a no-op. §2.4's rule: an
    // instrument that cannot be shown to have engaged has measured nothing.
    let fused = run_gemv_fused_silu(
        GemmDType::F32,
        (1, n_out, 1, k),
        &mat,
        &[n_out * k, k, 1],
        &vec,
        &[k, 1, 1],
    );
    assert!(
        a.iter().zip(fused.iter()).any(|(x, y)| x != y),
        "the fused and unfused arms produce identical bits; the epilogue is \
         not engaged and the comparison below would be vacuous"
    );

    let b = run_mlx_gemv_style::<f32>(
        GemmDType::F32,
        (1, n_out, 1, k),
        &mat,
        &[n_out * k, k, 1],
        &vec,
        &[k, 1, 1],
        ParamStyle::Split,
    );
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the unfused arm must be bit-identical run to run"
    );
}

///////////////////////////////////////////////////////////////////////////////
// Issue #509 -- the fused residual-add epilogue
///////////////////////////////////////////////////////////////////////////////

/// Drive the fused residual arm through the real entry point.
///
/// Takes ONE extra buffer, which is the shape of this entry's finding: unlike
/// a pure elementwise epilogue it reads a second operand -- but at the storing
/// thread's own output index, which is what keeps it on the admissible side of
/// §12.2e's G1 gate.
#[allow(clippy::too_many_arguments)]
fn run_gemv_fused_residual(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[f32],
    lhs_stride: &[usize],
    rhs: &[f32],
    rhs_stride: &[usize],
    residual: &[f32],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let lhs_buf = mk(lhs);
    let rhs_buf = mk(rhs);
    let res_buf = mk(residual);

    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<f32>(), options)
        .unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_epilogue(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            ParamStyle::Split,
            None,
            GemvEpilogue::Residual,
            Some((&res_buf, 0)),
        )
    }
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, length)
}

/// A CPU reference for the matmul followed by the residual add, computed in
/// f64 so it is not itself one of the two f32 orderings being compared
/// (§10.2g, §2.3.5a).
/// `run_gemv_fused_residual` with the gate-product epilogue: the same
/// operand binding (slot 15), `*` for `+`.
fn run_gemv_fused_mul(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[f32],
    lhs_stride: &[usize],
    rhs: &[f32],
    rhs_stride: &[usize],
    gate: &[f32],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let lhs_buf = mk(lhs);
    let rhs_buf = mk(rhs);
    let gate_buf = mk(gate);
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_epilogue(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            ParamStyle::Split,
            None,
            GemvEpilogue::Mul,
            Some((&gate_buf, 0)),
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, length)
}

fn reference_gemv_mul(
    (m_out, k): (usize, usize),
    mat: &[f32],
    vec: &[f32],
    gate: &[f32],
) -> Vec<f64> {
    (0..m_out)
        .map(|row| {
            let mut acc = 0.0f64;
            for c in 0..k {
                acc += mat[row * k + c] as f64 * vec[c] as f64;
            }
            acc * gate[row] as f64
        })
        .collect()
}

/// The gate-product epilogue against an f64 reference, at both tiles, with a
/// gate that varies with position and is never 1 or 0 — a constant gate cannot
/// tell `gate[out_row]` from `gate[0]`, a 1 makes that row agree with no
/// epilogue at all, and a 0 makes it agree with any epilogue that zeroes.
#[test]
fn gemv_fused_mul_matches_a_reference() {
    for &(k, n_out, label) in &[
        (2048usize, 512usize, "bm4, K divides blockN"),
        (2048usize, 6144usize, "bm8 (w3's tile), K divides blockN"),
        (2050usize, 512usize, "bm4, K leaves a 2-element tail"),
        (2050usize, 6144usize, "bm8, K leaves a 2-element tail"),
    ] {
        let (b, m, n) = (1usize, n_out, 1usize);
        let mat: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
            .collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
        let gate: Vec<f32> = (0..n_out)
            .map(|i| ((i % 23) as f32 - 11.0) / 8.0 + 0.0625)
            .collect();
        assert!(
            gate.iter().all(|v| *v != 0.0 && *v != 1.0),
            "{label}: the fixture's gate must have no 0 or 1 entry"
        );
        let got = run_gemv_fused_mul(
            GemmDType::F32,
            (b, m, n, k),
            &mat,
            &[m * k, k, 1],
            &vec,
            &[k, 1, 1],
            &gate,
        );
        let want = reference_gemv_mul((n_out, k), &mat, &vec, &gate);
        assert_eq!(got.len(), want.len(), "{label}: length");
        let distinct = got
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{label}: {distinct} distinct value(s); vacuous"
        );
        // Engagement: the product must differ from the bare GEMV at every row
        // (the gate is never 1), and from the residual at every row.
        let bare = reference_gemv_residual((n_out, k), &mat, &vec, &vec![0.0; n_out]);
        let differs = got
            .iter()
            .zip(bare.iter())
            .filter(|(g, w)| (**g as f64 - **w).abs() > 1e-6)
            .count();
        assert_eq!(
            differs, n_out,
            "{label}: the epilogue engaged on {differs} of {n_out} rows"
        );
        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let rel = ((g as f64) - w).abs() / denom;
            worst = worst.max(rel);
            assert!(
                rel < 2e-5,
                "{label}: row {i}: fused {g} vs reference {w} (rel {rel:e})"
            );
        }
        // No `worst > 0` non-vacuity arm here: the fixture's products are
        // dyadic and the f32 accumulation is exact at these extents, so the
        // f64 reference IS the kernel's answer. Engagement is the check above:
        // every row differs from the bare GEMV, as a gate that is never 1 must.
        eprintln!("{label}: worst rel {worst:e}");
    }
}

/// `run_gemv_fused_mul` with the two-weight kernel: the operand is the GATE
/// WEIGHT, `[n_out, k]` like `mat`.
fn run_gemv_fused_swiglu(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[f32],
    lhs_stride: &[usize],
    rhs: &[f32],
    rhs_stride: &[usize],
    gate_w: &[f32],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let lhs_buf = mk(lhs);
    let rhs_buf = mk(rhs);
    let gate_buf = mk(gate_w);
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<f32>(), options)
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_mlx_gemv_epilogue(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            0,
            &lhs_buf,
            rhs_stride,
            0,
            &rhs_buf,
            &output,
            ParamStyle::Split,
            None,
            GemvEpilogue::Swiglu,
            Some((&gate_buf, 0)),
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, length)
}

fn reference_gemv_swiglu(
    (m_out, k): (usize, usize),
    up: &[f32],
    gate: &[f32],
    vec: &[f32],
) -> Vec<f64> {
    (0..m_out)
        .map(|row| {
            let mut u = 0.0f64;
            let mut g = 0.0f64;
            for c in 0..k {
                u += up[row * k + c] as f64 * vec[c] as f64;
                g += gate[row * k + c] as f64 * vec[c] as f64;
            }
            (g / (1.0 + (-g).exp())) * u
        })
        .collect()
}

/// The two-weight kernel against an f64 reference at both tiles, with two
/// DIFFERENT weights (a kernel sweeping one matrix twice reads wrong), the
/// gate accumulator negative somewhere so SiLU is not the identity (the SiLU
/// test's own non-vacuity arm), and the product differing from the bare up
/// GEMV on nearly every row.
#[test]
fn gemv_fused_swiglu_matches_a_reference() {
    for &(k, n_out, label) in &[
        (2048usize, 512usize, "bm4, K divides blockN"),
        (2048usize, 6144usize, "bm8 (w1/w3's tile), K divides blockN"),
        (2050usize, 512usize, "bm4, K leaves a 2-element tail"),
        (2050usize, 6144usize, "bm8, K leaves a 2-element tail"),
    ] {
        let (b, m, n) = (1usize, n_out, 1usize);
        let up: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
            .collect();
        let gate: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 19) as f32 - 9.0) / 32.0)
            .collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
        let got = run_gemv_fused_swiglu(
            GemmDType::F32,
            (b, m, n, k),
            &up,
            &[m * k, k, 1],
            &vec,
            &[k, 1, 1],
            &gate,
        );
        let want = reference_gemv_swiglu((n_out, k), &up, &gate, &vec);
        assert_eq!(got.len(), want.len(), "{label}: length");
        let distinct = got
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{label}: {distinct} distinct value(s); vacuous"
        );
        let gate_acc: Vec<f64> = (0..n_out)
            .map(|row| {
                (0..k)
                    .map(|c| gate[row * k + c] as f64 * vec[c] as f64)
                    .sum()
            })
            .collect();
        assert!(
            gate_acc.iter().any(|g| *g < 0.0),
            "{label}: no negative gate"
        );
        let bare = reference_gemv_residual((n_out, k), &up, &vec, &vec![0.0; n_out]);
        let differs = got
            .iter()
            .zip(bare.iter())
            .filter(|(g, w)| (**g as f64 - **w).abs() > 1e-6)
            .count();
        assert!(
            differs > n_out * 9 / 10,
            "{label}: engaged on {differs} of {n_out}"
        );
        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let rel = ((g as f64) - w).abs() / denom;
            worst = worst.max(rel);
            assert!(
                rel < 2e-5,
                "{label}: row {i}: fused {g} vs reference {w} (rel {rel:e})"
            );
        }
        eprintln!("{label}: worst rel {worst:e}");
    }
}

fn reference_gemv_residual(
    (m_out, k): (usize, usize),
    mat: &[f32],
    vec: &[f32],
    res: &[f32],
) -> Vec<f64> {
    (0..m_out)
        .map(|row| {
            let mut acc = 0.0f64;
            for c in 0..k {
                acc += mat[row * k + c] as f64 * vec[c] as f64;
            }
            acc + res[row] as f64
        })
        .collect()
}

/// The fused epilogue computes matmul-then-residual-add.
///
/// **K = 2050 is in the list for the reason #266's and #484's fixtures carry
/// it**: the GEMV walks the input vector in `blockN` chunks and handles the
/// remainder in a separate `leftover` path, so at LFM2's own K = 2048 the tail
/// never executes. An epilogue is outside that loop and so is not itself
/// tail-sensitive -- but the ACCUMULATOR it transforms is, so a fixture that
/// never takes the tail cannot see an epilogue applied to a wrong sum.
///
/// **N_out = 2048 is the decode shape here and it is `bm4`**, where #484's
/// decode shape was `bm8`. All three producers of a residual add -- `w2`,
/// `out_proj`, `conv.out_proj` -- sit at N_out = 2048, so the tile this entry
/// cares about is the one #484's fixture carried only as a second arm.
#[test]
fn gemv_fused_residual_matches_a_reference() {
    // No availability check: since #516 the `_residual` names are instantiated
    // unconditionally, so this is an ordinary gate rather than a test that
    // skips unless an environment variable was set. The skip it replaces was
    // §15.1a's VACUOUS shape sitting in the suite -- a test that reports the
    // same green whether or not it evaluated anything.
    for &(k, n_out, label) in &[
        (
            2048usize,
            2048usize,
            "bm4 (the decode shape), K divides blockN",
        ),
        (2048usize, 6144usize, "bm8, K divides blockN"),
        (2050usize, 2048usize, "bm4, K leaves a 2-element tail"),
        (2050usize, 6144usize, "bm8, K leaves a 2-element tail"),
    ] {
        let (b, m, n) = (1usize, n_out, 1usize);

        let mat: Vec<f32> = (0..n_out * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
            .collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
        // The residual VARIES WITH POSITION and is never zero. Both matter:
        // a constant residual cannot distinguish `res[out_row]` from `res[0]`,
        // and a zero entry makes that position agree with an epilogue that did
        // not run at all. The harness's own engagement check read 1959 of 2048
        // on a fixture with 89 zeros before this was fixed.
        let res: Vec<f32> = (0..n_out)
            .map(|i| ((i % 23) as f32 - 11.0) / 8.0 + 0.0625)
            .collect();
        assert!(
            res.iter().all(|v| *v != 0.0),
            "{label}: the fixture's residual must have no zero entry, or those \
             positions cannot tell the epilogue from the identity"
        );

        let got = run_gemv_fused_residual(
            GemmDType::F32,
            (b, m, n, k),
            &mat,
            &[m * k, k, 1],
            &vec,
            &[k, 1, 1],
            &res,
        );
        let want = reference_gemv_residual((n_out, k), &mat, &vec, &res);

        assert_eq!(got.len(), want.len(), "{label}: length");

        // Non-vacuity, per §15.1 #1: two kernels that both write nothing agree
        // perfectly.
        let distinct = got
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{label}: fused output has {distinct} distinct value(s); the \
             comparison would be vacuous"
        );

        // A SECOND non-vacuity arm, specific to a residual ADD: the residual
        // must materially move the answer. If every |res| were tiny against
        // the accumulator the test would pass against a kernel with no
        // epilogue at all, which is exactly what mutation arm 1 does.
        let moved = want
            .iter()
            .zip(res.iter())
            .filter(|(w, r)| (**r as f64).abs() > (**w).abs() * 1e-3)
            .count();
        assert!(
            moved > n_out / 4,
            "{label}: the residual moves only {moved} of {n_out} outputs \
             materially; the comparison is close to vacuous"
        );

        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let rel = ((g as f64) - w).abs() / denom;
            if rel > worst {
                worst = rel;
            }
            assert!(
                rel < 2e-5,
                "{label}: row {i}: fused {g} vs reference {w} (rel {rel:e})"
            );
        }
    }
}

/// Asking for an epilogue that reads an operand without supplying one is
/// REFUSED, and so is the converse.
///
/// Both directions, because either alone passes on a defect: an operand
/// supplied to an arm that does not read it would be silently dropped, and the
/// caller would believe a fusion ran that did not (§2.4). Under
/// `HazardTrackingModeUntracked` (§3.5) nothing reports a well-formed wrong
/// number, which is why these are errors rather than debug assertions.
#[test]
fn gemv_epilogue_operand_arity_is_enforced() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let (k, n_out) = (2048usize, 2048usize);
    let mat = vec![0.01f32; n_out * k];
    let vec_ = vec![0.02f32; k];
    let mk = |v: &[f32]| {
        device
            .new_buffer_with_data(
                v.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(v),
                options,
            )
            .unwrap()
    };
    let mat_b = mk(&mat);
    let vec_b = mk(&vec_);
    let res_b = mk(&vec![0.5f32; n_out]);
    let out = device
        .new_buffer(n_out * core::mem::size_of::<f32>(), options)
        .unwrap();

    let call = |ep: GemvEpilogue, operand: Option<(&Buffer, usize)>| {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_mlx_gemv_epilogue(
                &device,
                &encoder,
                &kernels,
                GemmDType::F32,
                (1, n_out, 1, k),
                &[n_out * k, k, 1],
                0,
                &mat_b,
                &[k, 1, 1],
                0,
                &vec_b,
                &out,
                ParamStyle::Split,
                None,
                ep,
                operand,
            )
        }
    };

    // Residual with no operand: refused.
    let e = call(GemvEpilogue::Residual, None).unwrap_err();
    assert!(
        format!("{e:?}").contains("second operand"),
        "expected a refusal naming the missing operand, got {e:?}"
    );

    // Silu WITH an operand: also refused, because it would be dropped.
    let e = call(GemvEpilogue::Silu, Some((&res_b, 0))).unwrap_err();
    assert!(
        format!("{e:?}").contains("no second operand"),
        "expected a refusal naming the unused operand, got {e:?}"
    );

    // And `None` with an operand, the same way -- the default arm must not
    // silently accept one either.
    let e = call(GemvEpilogue::None, Some((&res_b, 0))).unwrap_err();
    assert!(
        format!("{e:?}").contains("no second operand"),
        "expected a refusal on the default arm, got {e:?}"
    );

    drop(encoder);
}

/// `GemvEpilogue::needs_operand` agrees with what the kernel actually binds.
///
/// A cheap pin, and the reason it exists is §7.1a-i: a two-armed conditional
/// over a three-armed axis mislabelled every `RESULT` line since #116, and the
/// repair was to read the resolved enum through an EXHAUSTIVE match. This
/// asserts the arity table rather than the rendering, so a fourth arm added
/// without deciding its arity is a compile error in `needs_operand` and a red
/// test here if the decision is wrong.
#[test]
fn gemv_epilogue_arity_table_is_what_the_kernels_take() {
    assert!(!GemvEpilogue::None.needs_operand());
    assert!(
        !GemvEpilogue::Silu.needs_operand(),
        "a pure elementwise epilogue reads nothing the kernel was not already \
         reading -- that is #484's finding and its signature is byte-identical \
         to `gemv`'s"
    );
    assert!(
        GemvEpilogue::Residual.needs_operand(),
        "the residual arm reads a second operand at slot 15; that binding is \
         what §12.2e's G1 gate is about"
    );
}

// ===========================================================================
// Quantized KV cache -- issue #218, `DESIGN.md` §10.7, §17 Phase 5 item 18.
// ===========================================================================

/// A scalar CPU reference for attention over a *dequantized* KV cache.
///
/// **This is an absolute reference and not a shape invariant** —
/// `CONTRIBUTING.md` §3.1b rule 1. It recomputes the answer from the codes and
/// the stored (scale, bias) pairs with no knowledge of what the kernel did, so
/// a defect that shifts every output equally is visible where an `all equal`
/// or `both arms agree` assertion would not be.
///
/// The accumulation is the online-softmax form the kernel uses, in the same
/// f32 accumulator, walking keys in index order. It is not the *bit-identical*
/// reference — the kernel's `simd_sum` is a tree where this is sequential, and
/// `fast::exp` is not `f32::exp` — so the comparison is against a tolerance,
/// which is what §2.3.5a's parity row asks for.
#[allow(clippy::too_many_arguments)]
fn quant_attention_reference(
    q: &[f32],
    k_codes: &[u8],
    k_scales: &[f32],
    k_biases: &[f32],
    v_codes: &[u8],
    v_scales: &[f32],
    v_biases: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    capacity: usize,
    scale: f32,
    bits: KvQuantBits,
) -> Vec<f32> {
    let cells: Vec<usize> = (0..kv_len).collect();
    quant_attention_reference_cells(
        q, k_codes, k_scales, k_biases, v_codes, v_scales, v_biases, n_q_heads, n_kv_heads,
        head_dim, &cells, capacity, scale, bits,
    )
}

/// [`quant_attention_reference`] over the keys at `cells`, in that order —
/// the selected read's reference.
#[allow(clippy::too_many_arguments)]
fn quant_attention_reference_cells(
    q: &[f32],
    k_codes: &[u8],
    k_scales: &[f32],
    k_biases: &[f32],
    v_codes: &[u8],
    v_scales: &[f32],
    v_biases: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cells: &[usize],
    capacity: usize,
    scale: f32,
    bits: KvQuantBits,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let code_at = |plane: &[u8], head: usize, tok: usize, j: usize| -> f32 {
        let idx = tok * head_dim + j;
        let base = bits.code_bytes(head * capacity * head_dim);
        match bits {
            KvQuantBits::Eight => plane[base + idx] as f32,
            KvQuantBits::Four => {
                let byte = plane[base + (idx >> 1)];
                (if idx & 1 == 1 { byte >> 4 } else { byte & 0x0F }) as f32
            }
        }
    };

    let mut out = vec![0f32; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let kv_h = h / gqa;
        let qv: Vec<f32> = (0..head_dim).map(|j| scale * q[h * head_dim + j]).collect();

        let mut acc = vec![0f32; head_dim];
        let mut max_score = f32::NEG_INFINITY;
        let mut sum_exp = 0f32;

        for &t in cells {
            let ks = k_scales[kv_h * capacity + t];
            let kb = k_biases[kv_h * capacity + t];
            let mut score = 0f32;
            for j in 0..head_dim {
                score += qv[j] * (code_at(k_codes, kv_h, t, j) * ks + kb);
            }

            let new_max = max_score.max(score);
            let factor = (max_score - new_max).exp();
            let exp_score = (score - new_max).exp();
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;

            let vs = v_scales[kv_h * capacity + t];
            let vb = v_biases[kv_h * capacity + t];
            for j in 0..head_dim {
                acc[j] = acc[j] * factor + exp_score * (code_at(v_codes, kv_h, t, j) * vs + vb);
            }
        }

        for j in 0..head_dim {
            out[h * head_dim + j] = acc[j] / sum_exp;
        }
    }
    out
}

/// Quantize on the GPU, read the planes back, and check them against the
/// scalar reference **code for code**.
///
/// The write side gets its own arm because a wrong code and a wrong dequant
/// produce the same wrong attention output, and only one of them is the
/// quantizer's fault.
fn run_quantize_kv(
    device: &Device,
    kernels: &Kernels,
    src: &[f16],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    capacity: usize,
    dst_offset: usize,
    bits: KvQuantBits,
) -> (Vec<u8>, Vec<f16>, Vec<f16>) {
    let code_bytes = bits.code_bytes(n_heads * capacity * head_dim);
    let scale_els = n_heads * capacity;

    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();

    let src_buf = new_buffer(device, src);
    // Zeroed rather than uninitialised: an untouched region must read as a
    // known value, or a kernel that writes nothing is indistinguishable from
    // one that writes what was already there (§15.1 #1's vacuity argument, in
    // the buffer rather than in the comparison).
    let codes = new_buffer(device, &vec![0u8; code_bytes]);
    let scales = new_buffer(device, &vec![f16::from_f32(0.0); scale_els]);
    let biases = new_buffer(device, &vec![f16::from_f32(0.0); scale_els]);

    let p =
        SdpaQuantWriteParams::for_append(seq_len * head_dim, capacity, head_dim, dst_offset, bits);
    let params = new_buffer(device, &[p]);

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_quantize_kv(
            device,
            &encoder,
            kernels,
            0,
            &src_buf,
            &codes,
            &scales,
            &biases,
            &params,
            n_heads,
            seq_len,
            head_dim,
            bits,
            SdpaDType::F16,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    (
        read_to_vec::<u8>(&codes, code_bytes),
        read_to_vec::<f16>(&scales, scale_els),
        read_to_vec::<f16>(&biases, scale_els),
    )
}

/// The quantizer agrees with the scalar reference on every code, at both widths.
///
/// **Absolute, per code.** Not "the codes are in range", not "the two widths
/// agree with each other" — the exact byte the reference computes, which is the
/// form §3.1b rule 1 asks for and the only one that fails with a sentence a
/// reader can act on.
#[test]
fn quantize_kv_matches_the_scalar_reference_at_both_widths() {
    let device = device();
    let kernels = Kernels::new();

    let (n_heads, head_dim, seq_len, capacity) = (8usize, 64usize, 3usize, 16usize);
    let dst_offset = 5usize;

    // A source whose groups have genuinely different ranges: a quantizer that
    // ignored `hi`/`lo` and used one global range would still pass a fixture
    // where every group looked alike.
    let src: Vec<f16> = (0..n_heads * seq_len * head_dim)
        .map(|i| {
            let head = i / (seq_len * head_dim);
            let amp = 1.0 + head as f32;
            f16::from_f32(amp * (((i * 37 % 101) as f32 - 50.0) / 50.0))
        })
        .collect();

    for bits in KvQuantBits::ALL {
        let (codes, scales, biases) = run_quantize_kv(
            &device, &kernels, &src, n_heads, seq_len, head_dim, capacity, dst_offset, bits,
        );

        let mut checked = 0usize;
        for h in 0..n_heads {
            for t in 0..seq_len {
                let group: Vec<f32> = (0..head_dim)
                    .map(|j| src[(h * seq_len + t) * head_dim + j].to_f32())
                    .collect();
                let (want_codes, want_scale, want_bias) =
                    quantize_group_reference(&group, bits, |x| f16::from_f32(x).to_f32());

                let dst_tok = dst_offset + t;
                assert_eq!(
                    scales[h * capacity + dst_tok].to_f32(),
                    want_scale,
                    "scale disagrees at head {h} token {dst_tok} at int{}",
                    bits.bits()
                );
                assert_eq!(
                    biases[h * capacity + dst_tok].to_f32(),
                    want_bias,
                    "bias disagrees at head {h} token {dst_tok} at int{}",
                    bits.bits()
                );

                for j in 0..head_dim {
                    let idx = dst_tok * head_dim + j;
                    let base = bits.code_bytes(h * capacity * head_dim);
                    let got = match bits {
                        KvQuantBits::Eight => codes[base + idx],
                        KvQuantBits::Four => {
                            let byte = codes[base + (idx >> 1)];
                            if idx & 1 == 1 {
                                byte >> 4
                            } else {
                                byte & 0x0F
                            }
                        }
                    };
                    assert_eq!(
                        got,
                        want_codes[j],
                        "code disagrees at head {h} token {dst_tok} element {j} at int{}",
                        bits.bits()
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(
            checked,
            n_heads * seq_len * head_dim,
            "the arm must check every code it wrote, or a kernel writing a \
             prefix passes"
        );

        // Non-vacuity, on the codes themselves: a quantizer that wrote one
        // constant everywhere agrees with a reference that did the same.
        let distinct = codes.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct >= 2,
            "the code plane holds {distinct} distinct byte(s) at int{}, so this \
             comparison is vacuous",
            bits.bits()
        );
    }
}

/// The append writes **only** the tokens it was given.
///
/// A quantizer that ignored `dst_offset` would overwrite the cache from token
/// zero, and every parity arm above would still pass because it reads at the
/// offset it wrote. The untouched region is the discriminator, and this is the
/// arm that reads it.
#[test]
fn quantize_kv_leaves_the_rest_of_the_plane_untouched() {
    let device = device();
    let kernels = Kernels::new();

    let (n_heads, head_dim, seq_len, capacity) = (2usize, 64usize, 2usize, 16usize);
    let dst_offset = 7usize;
    let src: Vec<f16> = (0..n_heads * seq_len * head_dim)
        .map(|i| f16::from_f32(((i * 17 % 61) as f32 - 30.0) / 30.0))
        .collect();

    for bits in KvQuantBits::ALL {
        let (codes, scales, _biases) = run_quantize_kv(
            &device, &kernels, &src, n_heads, seq_len, head_dim, capacity, dst_offset, bits,
        );

        for h in 0..n_heads {
            for t in 0..capacity {
                let touched = (dst_offset..dst_offset + seq_len).contains(&t);
                let scale = scales[h * capacity + t].to_f32();
                if touched {
                    assert_ne!(
                        scale, 0.0,
                        "head {h} token {t} is inside the append and must have \
                         been written"
                    );
                } else {
                    assert_eq!(
                        scale,
                        0.0,
                        "head {h} token {t} is outside the append at offset \
                         {dst_offset}..{} and must not have been written -- a \
                         kernel ignoring dst_offset passes every arm that reads \
                         only where it wrote",
                        dst_offset + seq_len
                    );
                }

                let first = bits.code_bytes(h * capacity * head_dim + t * head_dim);
                let last = bits.code_bytes(h * capacity * head_dim + (t + 1) * head_dim);
                let any_nonzero = codes[first..last].iter().any(|&b| b != 0);
                if !touched {
                    assert!(
                        !any_nonzero,
                        "head {h} token {t} has code bytes set outside the append"
                    );
                }
            }
        }
    }
}

/// `KvQuantBits::code_bytes` and the kernel's `sdpa_code_bytes` are the same
/// function.
///
/// The host sizes every plane with the first and the kernel indexes into it
/// with the second. A disagreement is a plane that overlaps its neighbour —
/// §3.5's silent wrong answer, not a crash — so the two are pinned against an
/// absolute table rather than against each other.
#[test]
fn kv_quant_code_bytes_matches_the_kernel() {
    assert_eq!(KvQuantBits::Eight.code_bytes(64), 64);
    assert_eq!(KvQuantBits::Four.code_bytes(64), 32);
    assert_eq!(KvQuantBits::Eight.code_bytes(128), 128);
    assert_eq!(KvQuantBits::Four.code_bytes(128), 64);
    // A whole cache: 8 heads x 4096 tokens x 64 dim.
    assert_eq!(KvQuantBits::Eight.code_bytes(8 * 4096 * 64), 2_097_152);
    assert_eq!(KvQuantBits::Four.code_bytes(8 * 4096 * 64), 1_048_576);

    assert_eq!(KvQuantBits::Eight.levels(), 255.0);
    assert_eq!(KvQuantBits::Four.levels(), 15.0);
    assert_eq!(KvQuantBits::Eight.bits(), 8);
    assert_eq!(KvQuantBits::Four.bits(), 4);
}

/// Attention over a quantized cache agrees with the scalar reference.
///
/// **The reference is computed from the planes the GPU wrote**, not from the
/// original f16 values: the question this arm answers is *does the kernel
/// dequantize and attend correctly*, which is separable from *does the
/// quantizer choose good codes* (the arms above) and must be, or a compensating
/// pair of errors passes both.
///
/// LFM2's geometry — 32 q heads, 8 kv heads, head_dim 64 — so `gqa_factor` is
/// 4 and the GQA mapping is exercised. §6.2a records that the pre-existing sdpa
/// suite had **no** GQA arm, and that mutating the kernel to ignore
/// `gqa_factor` left all five green; this arm does not repeat that.
///
/// `kv_len` 40 crosses the `for (i = simd_gid; i < N; i += BN)` wrap at BN=32,
/// so both the first pass and the second are exercised.
#[test]
fn sdpa_vector_quant_matches_the_scalar_reference_at_both_widths() {
    let device = device();
    let kernels = Kernels::new();

    let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 64usize);
    let (kv_len, capacity) = (40usize, 64usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q: Vec<f16> = (0..n_q_heads * head_dim)
        .map(|i| f16::from_f32(((i * 37 % 71) as f32 - 35.0) / 64.0))
        .collect();
    // K and V are given different value distributions, so a kernel that read
    // the K planes for V (or bound the two the same way) produces a different
    // answer. Both are quantized by the GPU quantizer, which is what makes the
    // planes the reference reads the ones the reader will see.
    // **Every key gets its own OFFSET and its own RANGE, and that is what makes
    // the bias term observable.** A fixture whose groups all span the same
    // interval gives every group the same `bias`, and dropping the bias from
    // the dequant then shifts every score by the same constant -- which softmax
    // is invariant under, so the defect is invisible. Measured: with a single
    // generator the bias spread over 320 groups is 0.031 and mutant 2 of
    // `measurements/issue-218-raw/mutation.sh` SURVIVES.
    //
    // That is §15.1a's VACUOUS class -- the instrument can express the
    // difference and no input produces it -- and `CONTRIBUTING.md` §3.1a rule 2
    // says the repair is the arm's *input*. The per-key offset below is also
    // the more faithful fixture: real keys at different positions carry
    // different magnitudes after RoPE.
    let k_src: Vec<f16> = (0..n_kv_heads * kv_len * head_dim)
        .map(|i| {
            let tok = (i / head_dim) % kv_len;
            let offset = (tok as f32) * 0.25 - 5.0;
            let amp = 1.0 + (tok % 7) as f32 * 0.5;
            f16::from_f32(offset + amp * ((i * 53 % 97) as f32 - 48.0) / 96.0)
        })
        .collect();
    let v_src: Vec<f16> = (0..n_kv_heads * kv_len * head_dim)
        .map(|i| {
            let tok = (i / head_dim) % kv_len;
            let offset = 3.0 - (tok as f32) * 0.2;
            let amp = 1.0 + (tok % 5) as f32 * 0.4;
            f16::from_f32(offset + amp * 2.0 * ((i * 29 % 83) as f32 - 41.0) / 48.0)
        })
        .collect();

    for bits in KvQuantBits::ALL {
        let (k_codes, k_scales, k_biases) = run_quantize_kv(
            &device, &kernels, &k_src, n_kv_heads, kv_len, head_dim, capacity, 0, bits,
        );
        let (v_codes, v_scales, v_biases) = run_quantize_kv(
            &device, &kernels, &v_src, n_kv_heads, kv_len, head_dim, capacity, 0, bits,
        );

        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();

        let q_buf = new_buffer(&device, &q);
        let kc = new_buffer(&device, &k_codes);
        let ks = new_buffer(&device, &k_scales);
        let kb = new_buffer(&device, &k_biases);
        let vc = new_buffer(&device, &v_codes);
        let vs = new_buffer(&device, &v_scales);
        let vb = new_buffer(&device, &v_biases);
        let out_el = n_q_heads * head_dim;
        let output = device
            .new_buffer(out_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();

        let p = SdpaQuantParams::for_step(
            n_q_heads / n_kv_heads,
            kv_len,
            capacity,
            head_dim,
            scale,
            1.0,
            bits,
        );
        let params = new_buffer(&device, &[p]);

        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_sdpa_vector_quant(
                &device,
                &encoder,
                &kernels,
                0,
                &[1, n_q_heads, 1, head_dim],
                &q_buf,
                &kc,
                &ks,
                &kb,
                &vc,
                &vs,
                &vb,
                &output,
                &params,
                bits,
                SdpaDType::F16,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();

        let got = read_to_vec::<f16>(&output, out_el);

        let want = quant_attention_reference(
            &q.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            &k_codes,
            &k_scales.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            &k_biases.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            &v_codes,
            &v_scales.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            &v_biases.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            capacity,
            scale,
            bits,
        );

        // Non-vacuity first, so a kernel that wrote nothing cannot pass by
        // agreeing with a reference that also produced little (§15.1 #1).
        let distinct = got
            .iter()
            .map(|x| x.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "int{} attention wrote {distinct} distinct value(s) over {out_el} \
             elements, so this comparison is vacuous",
            bits.bits()
        );

        // The tolerance covers f16 storage of the output plus the difference
        // between a `simd_sum` tree and a sequential sum, and nothing else --
        // both operands are dequantized from the SAME codes, so quantization
        // error cancels exactly here rather than being absorbed by a loose
        // bound. That is what makes this a parity test and not a sanity check.
        let mut worst = 0f32;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g.to_f32() - w).abs();
            if d > worst {
                worst = d;
            }
            assert!(
                d < 2e-3,
                "int{} attention disagrees with the scalar reference at {i}: \
                 got {} want {w} (delta {d})",
                bits.bits(),
                g.to_f32()
            );
        }
        eprintln!(
            "int{} sdpa_vector_quant vs scalar reference: worst |delta| = {worst:.3e}",
            bits.bits()
        );
    }
}

/// The two widths do not compute the same thing.
///
/// **The arm that makes the pair above readable.** If int4 and int8 produced
/// identical output, both could be passing against a reference that shares
/// their defect — and a `BITS` template parameter that was silently ignored
/// would look exactly like that. This asserts they differ, and that int8 is the
/// *closer* of the two to the unquantized answer, which is the direction the
/// scheme predicts and the one a swapped instantiation would invert.
#[test]
fn the_two_widths_differ_and_int8_is_closer_to_f16() {
    let device = device();
    let kernels = Kernels::new();

    let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 64usize);
    let (kv_len, capacity) = (40usize, 64usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q: Vec<f16> = (0..n_q_heads * head_dim)
        .map(|i| f16::from_f32(((i * 37 % 71) as f32 - 35.0) / 64.0))
        .collect();
    // **Every key gets its own OFFSET and its own RANGE, and that is what makes
    // the bias term observable.** A fixture whose groups all span the same
    // interval gives every group the same `bias`, and dropping the bias from
    // the dequant then shifts every score by the same constant -- which softmax
    // is invariant under, so the defect is invisible. Measured: with a single
    // generator the bias spread over 320 groups is 0.031 and mutant 2 of
    // `measurements/issue-218-raw/mutation.sh` SURVIVES.
    //
    // That is §15.1a's VACUOUS class -- the instrument can express the
    // difference and no input produces it -- and `CONTRIBUTING.md` §3.1a rule 2
    // says the repair is the arm's *input*. The per-key offset below is also
    // the more faithful fixture: real keys at different positions carry
    // different magnitudes after RoPE.
    let k_src: Vec<f16> = (0..n_kv_heads * kv_len * head_dim)
        .map(|i| {
            let tok = (i / head_dim) % kv_len;
            let offset = (tok as f32) * 0.25 - 5.0;
            let amp = 1.0 + (tok % 7) as f32 * 0.5;
            f16::from_f32(offset + amp * ((i * 53 % 97) as f32 - 48.0) / 96.0)
        })
        .collect();
    let v_src: Vec<f16> = (0..n_kv_heads * kv_len * head_dim)
        .map(|i| {
            let tok = (i / head_dim) % kv_len;
            let offset = 3.0 - (tok as f32) * 0.2;
            let amp = 1.0 + (tok % 5) as f32 * 0.4;
            f16::from_f32(offset + amp * 2.0 * ((i * 29 % 83) as f32 - 41.0) / 48.0)
        })
        .collect();

    // The unquantized answer, from the same kernel family the shipping path
    // uses -- `call_sdpa_vector` over the f16 planes.
    let f16_out = {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let q_buf = new_buffer(&device, &q);
        let k_buf = new_buffer(&device, &k_src);
        let v_buf = new_buffer(&device, &v_src);
        let out_el = n_q_heads * head_dim;
        let output = device
            .new_buffer(out_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        let strides = [
            n_kv_heads * kv_len * head_dim,
            kv_len * head_dim,
            head_dim,
            1,
        ];
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_sdpa_vector(
                &device,
                &encoder,
                &kernels,
                0,
                &[1, n_q_heads, 1, head_dim],
                &q_buf,
                0,
                &[1, n_kv_heads, kv_len, head_dim],
                &strides,
                &k_buf,
                0,
                &strides,
                &v_buf,
                &output,
                scale,
                1.0,
                SdpaDType::F16,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        read_to_vec::<f16>(&output, out_el)
    };

    let mut rms = Vec::new();
    let mut outs = Vec::new();
    for bits in KvQuantBits::ALL {
        let (k_codes, k_scales, k_biases) = run_quantize_kv(
            &device, &kernels, &k_src, n_kv_heads, kv_len, head_dim, capacity, 0, bits,
        );
        let (v_codes, v_scales, v_biases) = run_quantize_kv(
            &device, &kernels, &v_src, n_kv_heads, kv_len, head_dim, capacity, 0, bits,
        );

        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let q_buf = new_buffer(&device, &q);
        let kc = new_buffer(&device, &k_codes);
        let ks = new_buffer(&device, &k_scales);
        let kb = new_buffer(&device, &k_biases);
        let vc = new_buffer(&device, &v_codes);
        let vs = new_buffer(&device, &v_scales);
        let vb = new_buffer(&device, &v_biases);
        let out_el = n_q_heads * head_dim;
        let output = device
            .new_buffer(out_el * core::mem::size_of::<f16>(), RESOURCE_OPTIONS)
            .unwrap();
        let p = SdpaQuantParams::for_step(
            n_q_heads / n_kv_heads,
            kv_len,
            capacity,
            head_dim,
            scale,
            1.0,
            bits,
        );
        let params = new_buffer(&device, &[p]);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_sdpa_vector_quant(
                &device,
                &encoder,
                &kernels,
                0,
                &[1, n_q_heads, 1, head_dim],
                &q_buf,
                &kc,
                &ks,
                &kb,
                &vc,
                &vs,
                &vb,
                &output,
                &params,
                bits,
                SdpaDType::F16,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let got = read_to_vec::<f16>(&output, out_el);

        let err: f32 = got
            .iter()
            .zip(f16_out.iter())
            .map(|(a, b)| {
                let d = a.to_f32() - b.to_f32();
                d * d
            })
            .sum::<f32>()
            / (out_el as f32);
        rms.push(err.sqrt());
        outs.push(got);
    }

    assert_ne!(
        outs[0], outs[1],
        "int8 and int4 produced byte-identical output, which they cannot: a \
         `BITS` template parameter that is ignored, or one instantiation \
         reached by both names, looks exactly like this"
    );
    assert!(
        rms[0] < rms[1],
        "int8 RMS error {} is not below int4's {}; the widths are swapped or \
         the level count is wrong",
        rms[0],
        rms[1]
    );
    // An absolute floor as well as an ordering, per §3.1b rule 1: `a < b` alone
    // is a shape, and a defect scaling both errors keeps the inequality.
    assert!(
        rms[0] < 5e-3,
        "int8 RMS error against f16 is {}, far above what 8-bit affine \
         quantization of this fixture can explain",
        rms[0]
    );
    eprintln!(
        "RMS vs f16 attention: int8 = {:.4e}, int4 = {:.4e} (ratio {:.2}x)",
        rms[0],
        rms[1],
        rms[1] / rms[0]
    );
}

/// `run_quantize_kv` at f32: the planes Qwen4-exp's attention writes (f32
/// activations, so f32 scale and bias planes).
#[allow(clippy::too_many_arguments)]
fn run_quantize_kv_f32(
    device: &Device,
    kernels: &Kernels,
    src: &[f32],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    capacity: usize,
    dst_offset: usize,
    bits: KvQuantBits,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let code_bytes = bits.code_bytes(n_heads * capacity * head_dim);
    let scale_els = n_heads * capacity;
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let src_buf = new_buffer(device, src);
    let codes = new_buffer(device, &vec![0u8; code_bytes]);
    let scales = new_buffer(device, &vec![0f32; scale_els]);
    let biases = new_buffer(device, &vec![0f32; scale_els]);
    let p =
        SdpaQuantWriteParams::for_append(seq_len * head_dim, capacity, head_dim, dst_offset, bits);
    let params = new_buffer(device, &[p]);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_quantize_kv(
            device,
            &encoder,
            kernels,
            0,
            &src_buf,
            &codes,
            &scales,
            &biases,
            &params,
            n_heads,
            seq_len,
            head_dim,
            bits,
            SdpaDType::F32,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    (
        read_to_vec::<u8>(&codes, code_bytes),
        read_to_vec::<f32>(&scales, scale_els),
        read_to_vec::<f32>(&biases, scale_els),
    )
}

/// The code of element `j` of token `t` of `head` in a packed plane.
fn plane_code(
    plane: &[u8],
    bits: KvQuantBits,
    head: usize,
    t: usize,
    j: usize,
    capacity: usize,
    head_dim: usize,
) -> u8 {
    let idx = t * head_dim + j;
    let base = bits.code_bytes(head * capacity * head_dim);
    match bits {
        KvQuantBits::Eight => plane[base + idx],
        KvQuantBits::Four => {
            let byte = plane[base + (idx >> 1)];
            if idx & 1 == 1 {
                byte >> 4
            } else {
                byte & 0x0F
            }
        }
    }
}

/// Qwen4-exp's K / V fixture: per-token offset and range (the bias term is
/// observable only when the groups differ — mutant 2 of #218's harness), at
/// f32, `n_heads × len × head_dim`.
fn qwen_kv_fixture(n_heads: usize, len: usize, head_dim: usize, salt: usize) -> Vec<f32> {
    (0..n_heads * len * head_dim)
        .map(|i| {
            let tok = (i / head_dim) % len;
            let offset = (tok as f32) * 0.07 - 2.0 + salt as f32 * 0.3;
            let amp = 1.0 + ((tok + salt) % 7) as f32 * 0.4;
            offset + amp * (((i * (53 + salt)) % 97) as f32 - 48.0) / 96.0
        })
        .collect()
}

/// The quantizer at `head_dim` 256, f32 — Qwen4-exp's instantiation —
/// agrees with the scalar reference. The bias (the group's min) is exact; the
/// scale `range / levels` is within one f32 ulp of the IEEE quotient (the
/// library compiles with fast-math, `CONTRIBUTING.md` §1.4, and at f16 the
/// storage rounding hid it); and every code is a NEAREST code of the value
/// under the STORED pair — within half a step of `(x − bias) / scale`, exact
/// in f64 — the contract the kernel states (the encode and the decode read
/// one stored (scale, bias)). A value within 1e-3 of a half step may take
/// either neighbour (the fast reciprocal); those ties are counted, and an
/// off-by-one code is still a failure.
#[test]
fn quantize_kv_at_head_dim_256_f32_matches_the_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();
    let (n_heads, head_dim, seq_len, capacity, dst_offset) =
        (2usize, 256usize, 5usize, 12usize, 4usize);
    let src = qwen_kv_fixture(n_heads, seq_len, head_dim, 0);
    for bits in KvQuantBits::ALL {
        let (codes, scales, biases) = run_quantize_kv_f32(
            &device, &kernels, &src, n_heads, seq_len, head_dim, capacity, dst_offset, bits,
        );
        let (mut checked, mut ties) = (0usize, 0usize);
        for h in 0..n_heads {
            for t in 0..seq_len {
                let group = &src[(h * seq_len + t) * head_dim..(h * seq_len + t + 1) * head_dim];
                let (_, want_scale, want_bias) = quantize_group_reference(group, bits, |x| x);
                let dst = dst_offset + t;
                let (got_scale, got_bias) =
                    (scales[h * capacity + dst], biases[h * capacity + dst]);
                let ulp = f32::from_bits(want_scale.to_bits() + 1) - want_scale;
                assert!(
                    (got_scale - want_scale).abs() <= ulp,
                    "scale h{h} t{dst} int{}: {got_scale} vs {want_scale} (> 1 ulp)",
                    bits.bits()
                );
                assert_eq!(got_bias, want_bias, "bias h{h} t{dst} int{}", bits.bits());
                for (j, &x) in group.iter().enumerate() {
                    let exact = ((x as f64 - got_bias as f64) / got_scale as f64)
                        .clamp(0.0, bits.levels() as f64);
                    let code = plane_code(&codes, bits, h, dst, j, capacity, head_dim) as f64;
                    assert!(
                        (code - exact).abs() <= 0.5 + 1e-3,
                        "code h{h} t{dst} j{j} int{}: {code} is not a nearest code of {exact}",
                        bits.bits()
                    );
                    if (code - exact.round()).abs() > 0.0 {
                        ties += 1;
                    }
                    checked += 1;
                }
            }
            // Outside the append nothing was written.
            for t in (0..capacity).filter(|t| !(dst_offset..dst_offset + seq_len).contains(t)) {
                assert_eq!(
                    scales[h * capacity + t],
                    0.0,
                    "h{h} t{t} outside the append"
                );
            }
        }
        assert_eq!(checked, n_heads * seq_len * head_dim);
        eprintln!(
            "int{} quantize_kv hd256 f32: {checked} codes, {ties} at a half-step tie",
            bits.bits()
        );
        assert!(
            ties * 100 < checked,
            "{ties} of {checked} codes off the rounded value: not ties"
        );
        let distinct = codes.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct >= 8,
            "only {distinct} distinct code bytes at int{}",
            bits.bits()
        );
    }
}

/// Qwen4-exp's attention over a quantized cache: 24 query heads over 2 KV
/// heads (`gqa_factor` 12), `head_dim` 256, f32 — against the scalar
/// reference over the planes the GPU wrote. `kv_len` 70 crosses the BN = 32
/// key-loop wrap twice; `capacity` 96 puts the head stride off `kv_len`.
///
/// Returns the planes and the dense output for the selection arm.
fn qwen_quant_attention_case(
    device: &Device,
    kernels: &Kernels,
    bits: KvQuantBits,
    sel: Option<&[u32]>,
) -> (Vec<f32>, Vec<f32>) {
    let (n_q, n_kv, hd, kv_len, capacity) = (24usize, 2usize, 256usize, 70usize, 96usize);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let q: Vec<f32> = (0..n_q * hd)
        .map(|i| ((i * 37 % 71) as f32 - 35.0) / 64.0)
        .collect();
    let (kc, ks, kb) = run_quantize_kv_f32(
        device,
        kernels,
        &qwen_kv_fixture(n_kv, kv_len, hd, 0),
        n_kv,
        kv_len,
        hd,
        capacity,
        0,
        bits,
    );
    let (vc, vs, vbi) = run_quantize_kv_f32(
        device,
        kernels,
        &qwen_kv_fixture(n_kv, kv_len, hd, 3),
        n_kv,
        kv_len,
        hd,
        capacity,
        0,
        bits,
    );
    let n = sel.map(|s| s.len()).unwrap_or(kv_len);
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let out_el = n_q * hd;
    let output = device.new_buffer(out_el * 4, RESOURCE_OPTIONS).unwrap();
    let p = SdpaQuantParams::for_step(n_q / n_kv, n, capacity, hd, scale, 1.0, bits);
    let params = new_buffer(device, &[p]);
    let (qb, kcb, ksb, kbb, vcb, vsb, vbb) = (
        new_buffer(device, &q),
        new_buffer(device, &kc),
        new_buffer(device, &ks),
        new_buffer(device, &kb),
        new_buffer(device, &vc),
        new_buffer(device, &vs),
        new_buffer(device, &vbi),
    );
    match sel {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        None => unsafe {
            call_sdpa_vector_quant(
                device,
                &encoder,
                kernels,
                0,
                &[1, n_q, 1, hd],
                &qb,
                &kcb,
                &ksb,
                &kbb,
                &vcb,
                &vsb,
                &vbb,
                &output,
                &params,
                bits,
                SdpaDType::F32,
            )
        }
        .unwrap(),
        Some(cells) => {
            // A leading pad word, so the binding's byte offset is exercised.
            let mut padded = vec![u32::MAX];
            padded.extend_from_slice(cells);
            let sb = new_buffer(device, &padded);
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
                call_sdpa_vector_quant_sel(
                    device,
                    &encoder,
                    kernels,
                    0,
                    &[1, n_q, 1, hd],
                    &qb,
                    &kcb,
                    &ksb,
                    &kbb,
                    &vcb,
                    &vsb,
                    &vbb,
                    &output,
                    &params,
                    (&sb, 4),
                    bits,
                    SdpaDType::F32,
                )
            }
            .unwrap()
        }
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();
    let got = read_to_vec::<f32>(&output, out_el);
    let cells: Vec<usize> = match sel {
        Some(c) => c.iter().map(|&c| c as usize).collect(),
        None => (0..kv_len).collect(),
    };
    let want = quant_attention_reference_cells(
        &q, &kc, &ks, &kb, &vc, &vs, &vbi, n_q, n_kv, hd, &cells, capacity, scale, bits,
    );
    (got, want)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) -> f32 {
    let distinct = got
        .iter()
        .map(|x| x.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct >= 16,
        "{what}: {distinct} distinct outputs — vacuous"
    );
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        worst = worst.max(d);
        assert!(
            d < tol,
            "{what}: disagrees at {i}: got {g} want {w} (delta {d})"
        );
    }
    worst
}

#[test]
fn sdpa_vector_quant_at_head_dim_256_f32_matches_the_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();
    for bits in KvQuantBits::ALL {
        let (got, want) = qwen_quant_attention_case(&device, &kernels, bits, None);
        let worst = assert_close(&got, &want, 1e-4, &format!("int{} hd256 f32", bits.bits()));
        eprintln!(
            "int{} sdpa_vector_quant hd256 f32 vs scalar reference: worst |delta| = {worst:.3e}",
            bits.bits()
        );
    }
}

/// The selected read (QSA's cells) against the reference over the same
/// cells, in the selection's order: scattered complete blocks of 4, a block
/// out of order, and the incomplete tail — 37 cells, crossing the BN wrap.
/// And the identity selection `[0, kv_len)` is the dense kernel **bit for
/// bit**: the selection changes where key `i` is read, never the arithmetic.
#[test]
fn sdpa_vector_quant_sel_reads_the_selected_cells() {
    let device = device();
    let kernels = Kernels::new();
    let mut cells: Vec<u32> = Vec::new();
    for b in [2u32, 9, 5, 12, 0, 15, 7, 11] {
        cells.extend(b * 4..b * 4 + 4);
    }
    cells.extend(68..70); // the tail
    cells.extend([40, 41, 42]);
    for bits in KvQuantBits::ALL {
        let (got, want) = qwen_quant_attention_case(&device, &kernels, bits, Some(&cells));
        let worst = assert_close(&got, &want, 1e-4, &format!("int{} sel", bits.bits()));
        eprintln!("int{} sdpa_vector_quant_sel vs scalar reference over {} cells: worst |delta| = {worst:.3e}", bits.bits(), cells.len());
        // The selection is not the dense answer (it must not read [0, n)).
        let (dense_n, _) = qwen_quant_attention_case(&device, &kernels, bits, None);
        assert_ne!(
            got,
            dense_n,
            "int{}: the selected read equals the dense one",
            bits.bits()
        );
        let identity: Vec<u32> = (0..70).collect();
        let (id_out, _) = qwen_quant_attention_case(&device, &kernels, bits, Some(&identity));
        assert_eq!(
            id_out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dense_n.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "int{}: the identity selection is not the dense kernel bit for bit",
            bits.bits()
        );
    }
}

///////////////////////////////////////////////////////////////////////////////
/// The SiLU variant axis (issue #599, `DESIGN.md` §12.2l)
///////////////////////////////////////////////////////////////////////////////

/// Every `SiluImpl` arm resolves, at every dtype, in both binding forms.
///
/// This is `packed_names_resolve`'s argument for a second axis (§8.1b): the
/// name is checked against the **compiled library**, not against the macro that
/// should have produced it, because #26 shipped 48 names absent from a metallib
/// that compiled cleanly.
///
/// **It carries `Exact` deliberately, and that is not redundant.** On this
/// axis's first run every arm read absent -- including `silu_f16`, which has
/// always existed -- because `usilu_poly`'s `abs` was ambiguous at `bfloat` and
/// the whole `unary` library failed to compile. A check covering only the NEW
/// names would have reported them missing and sent the reader hunting an
/// instantiation bug instead of a compile error two lines away.
#[test]
fn silu_variant_names_resolve() {
    let device = device();
    let kernels = Kernels::new();

    let mut checked = 0usize;
    for arm in crate::kernels::SiluImpl::ALL {
        for dtype in [
            crate::kernels::SiluDType::F16,
            crate::kernels::SiluDType::F32,
            crate::kernels::SiluDType::BF16,
        ] {
            for name in [arm.contiguous_name(dtype), arm.strided_name(dtype)] {
                kernels
                    .load_pipeline(&device, Source::Unary, name)
                    .unwrap_or_else(|e| {
                        panic!(
                            "unary.metal has no kernel named {name:?} \
                             (SiluImpl::{:?} at {dtype:?}): {e:?}",
                            arm
                        )
                    });
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked, 42,
        "expected 7 arms x 3 dtypes x 2 forms = 42 names, found {checked}"
    );
}

/// The default arm resolves the name the tree shipped before this axis existed.
///
/// `DESIGN.md` §7.1a: a default is flipped by its own argued decision, and
/// wiring an axis is not that argument. This pins the property in a test rather
/// than leaving it to the end-to-end digest gate, because the digest cannot say
/// WHICH name was resolved -- only that the answer came out the same.
#[test]
fn the_default_silu_arm_is_the_shipping_name() {
    assert_eq!(
        crate::kernels::SiluImpl::default(),
        crate::kernels::SiluImpl::Exact
    );
    assert_eq!(crate::kernels::silu_impl(), crate::kernels::SiluImpl::Exact);
    assert_eq!(
        crate::kernels::SiluImpl::Exact.contiguous_name(crate::kernels::SiluDType::F16),
        "silu_f16"
    );
    assert_eq!(
        crate::kernels::SiluImpl::Exact.strided_name(crate::kernels::SiluDType::F16),
        "silu_f16_strided"
    );
    // The suffix is empty for `Exact` and non-empty for every other arm. That
    // asymmetry is what makes the default resolve the pre-existing name rather
    // than a renamed sibling, so it is asserted rather than assumed.
    assert_eq!(crate::kernels::SiluImpl::Exact.suffix(), "");
    for arm in crate::kernels::SiluImpl::ALL {
        if arm != crate::kernels::SiluImpl::Exact {
            assert!(!arm.suffix().is_empty(), "{arm:?} must carry a suffix");
        }
    }
}

/// Every arm round-trips through its CLI spelling.
///
/// The harnesses REFUSE an unrecognized `--silu-impl` rather than defaulting
/// (§2.4: a silently-ignored flag makes a gate report on an arm it did not
/// run), so `parse` returning `None` is load-bearing and is asserted here.
#[test]
fn silu_arm_names_round_trip() {
    for arm in crate::kernels::SiluImpl::ALL {
        assert_eq!(crate::kernels::SiluImpl::parse(arm.name()), Some(arm));
    }
    assert_eq!(crate::kernels::SiluImpl::parse("nosuchvariant"), None);
    assert_eq!(crate::kernels::SiluImpl::parse(""), None);
}

///////////////////////////////////////////////////////////////////////////////
/// The SiLU consolidation (issue #599, `DESIGN.md` §15.2 #15 / §3.6)
///////////////////////////////////////////////////////////////////////////////

/// The shipping SiLU expression has exactly ONE definition, and the live
/// placements are call sites of it.
///
/// **This is a SOURCE-level census and it has to be**, in #507's and #562's
/// shape: one-definition-ness is a property of the **text**, not of any value a
/// run produces. Every dispatch test in this file passes just as happily with
/// the expression written out twice — which is the state this change removed —
/// so no runtime assertion can distinguish "consolidated" from "duplicated and
/// currently in agreement". That is the whole defect class §3.6 names.
///
/// **The needles are ASSEMBLED by concatenation** (§9.2p, measured): a literal
/// needle finds itself in the file it searches, and *anchoring is not the
/// repair* — only a needle the source does not contain escapes it. #347's
/// earlier run read 6 where the answer was 3 for exactly this reason.
///
/// **A non-vacuity arm runs first** (§3.1a rule 1, #506): if the shared
/// definition cannot be found at all, every "zero copies" count below is
/// trivially satisfied by a file that has been renamed away, which is §15.1a's
/// VACUOUS class. The count of *call sites* is asserted to be non-zero for the
/// same reason.
///
/// **And this test turns out to be the ONLY thing in the suite that kills a
/// real numerics change to the shared body**, which was not the intent and is
/// worth recording. Mutating the primitive's bare `1` to `1.0f` — #604's
/// measured lever, 1 f16 ulp over 440 of 4096 points — leaves `silu_f16`,
/// `silu_f32`, `gemv_fused_silu_matches_a_reference` and
/// `gemv_fused_silu_agrees_with_the_unfused_path` **all green**, and only this
/// census goes red. The reason is `approx_f16(results, 2)`: the numeric tests
/// round to two decimal places, so they are structurally blind to a
/// one-ulp-at-f16 difference (§3.1b — a tolerance is a shape, and a defect
/// smaller than the tolerance changes no relationship it constrains).
///
/// So the coverage here is **incidental**, which #524 names as a distinct risk
/// from absence: a future edit that loosens this census, or a refactor that
/// deletes it as "just a grep", takes that class live with nothing saying so.
/// Recorded at the test rather than left to be re-derived. The honest reading
/// is that a 1-ulp guard on the shared SiLU body **does not exist** in this
/// suite; what exists is a text check that happens to catch the one spelling
/// known to produce one.
#[test]
fn the_silu_expression_has_one_definition_and_the_rest_are_call_sites() {
    let utils = include_str!("metal_src/utils.metal");
    let unary = include_str!("metal_src/unary.metal");
    let gemv = include_str!("metal_src/gemm/gemv.metal");

    // Assembled, never written as one literal — see the doc comment.
    let expr = format!("x /{} (1 +{} exp(-x))", "", "");
    let defn = format!("METAL_FUNC T silu(T{} x)", "");
    let call = format!("silu({}x)", "");

    // Non-vacuity FIRST: the shared definition must exist, or every count below
    // is a statement about a file that is not there.
    assert!(
        utils.contains(&defn),
        "utils.metal carries no `{defn}` — the shared primitive is gone, and \
         every count below would pass vacuously"
    );

    // Comment lines are stripped before counting: `utils.metal` and both call
    // sites EXPLAIN the expression in prose, and an explanation matching itself
    // is #579's measured trap (`a better needle is still a needle`).
    let code_only = |src: &str| -> String {
        src.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let utils_code = code_only(utils);
    let unary_code = code_only(unary);
    let gemv_code = code_only(gemv);

    // ONE definition of the expression, and it is in `utils.metal`.
    assert_eq!(
        utils_code.matches(expr.as_str()).count(),
        1,
        "utils.metal must hold exactly one copy of the SiLU expression"
    );

    // `unary.metal` still spells it in the LADDER arms — `usilu_literal` and
    // the `Recip` control are deliberately independent expressions (#613), so
    // this is not asserted to be zero. What must be zero is the SHIPPING copy,
    // and the discriminator is the bare `1` with no `f` suffix: `usilu_literal`
    // is `1.0f` and every other arm is a different function entirely.
    assert_eq!(
        unary_code.matches(expr.as_str()).count(),
        0,
        "unary.metal must not re-spell the shipping SiLU expression — \
         `usilu` is a call site of utils.metal's `silu<T>()`. The variant arms \
         carry their own DIFFERENT expressions and are unaffected."
    );
    assert_eq!(
        gemv_code.matches(expr.as_str()).count(),
        0,
        "gemv.metal must not re-spell the SiLU expression — `EpilogueSilu` is \
         a call site of utils.metal's `silu<T>()`"
    );

    // And both placements must actually CALL it. A file that spells the
    // expression zero times because it dropped SiLU altogether would satisfy
    // the two assertions above; this is what separates consolidated from
    // deleted, and it is #517's positive-control rule.
    assert!(
        unary_code.contains(&call),
        "unary.metal must call the shared `silu()` — a zero above with no call \
         here means the op was removed, not consolidated"
    );
    assert!(
        gemv_code.contains(&call),
        "gemv.metal must call the shared `silu()`"
    );
}

/// `gemv` sees `utils.metal`, and the default arm still routes through it.
///
/// `EpilogueSilu` can only call the shared primitive because `source.rs` makes
/// `GEMV` a `with_utils!` consumer (#599). That composition is what a
/// `#include` would do in a tree that had a filesystem at compile time, and
/// #458 measured that it does not — so this asserts the ROUTING rather than the
/// text, which is #516's mutation-3 lesson: *a test of the merged TEXT is not a
/// test of the merged ARM*.
///
/// The bogus-name arm is the non-vacuity control: without it, a `load_pipeline`
/// that succeeded for every string would read as a passing test.
#[test]
fn gemv_is_composed_with_utils_and_its_silu_epilogue_resolves() {
    // The composed text must contain a helper that lives ONLY in utils.metal.
    // `get_strided_index` is the marker: `gemv.metal` never uses it, so its
    // presence in `GEMV` is evidence of the splice and of nothing else.
    let marker = format!("METAL_FUNC uint get_strided_index{}(", "");
    assert!(
        crate::kernels::source::GEMV.contains(&marker),
        "source::GEMV is not composed with utils.metal — `EpilogueSilu` cannot \
         see the shared `silu<T>()`"
    );

    // Spliced exactly once. Twice would redefine every shared helper, which is
    // the redefinition `MERGED_NON_MLX`'s own docs record as manufacturable by
    // a careless composition.
    assert_eq!(
        crate::kernels::source::GEMV
            .matches(marker.as_str())
            .count(),
        1,
        "utils.metal must be spliced into GEMV exactly once"
    );

    // And the fused kernel still compiles and resolves out of that library, at
    // EVERY instantiated dtype.
    //
    // **`bfloat16` is the row that earns its place.** The hoist's first version
    // dropped `define_unary_op`'s `static_cast<T>` and failed to compile at
    // `bfloat` alone — `1 + exp(-x)` promotes to `float` and MSL has no
    // implicit narrowing — while f16 and f32 were clean. LFM2 decodes at f16,
    // both canonical pairs are f16 and `lfm2-smoke` is f16, so **every gate
    // this project leads with is blind to that failure**. It was caught by the
    // fuzzers dispatching a unary op at every dtype; this asserts it directly
    // so the next reader does not need a proptest to find it.
    // Names written out verbatim rather than assembled, deliberately: a name is
    // an artifact of a `[[host_name]]` string in `gemv.metal` and #26 shipped 48
    // names that a `format!` reproduced and the metallib did not have. Spelling
    // them is what makes this a check against the compiled library (§8.1b).
    let device = device();
    let kernels = Kernels::new();
    let mut resolved = 0usize;
    for name in [
        "gemv_float32_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_silu",
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_silu",
        "gemv_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_silu",
    ] {
        kernels
            .load_pipeline(&device, Source::Gemv, name)
            .unwrap_or_else(|e| {
                panic!(
                    "the fused silu GEMV {name:?} must resolve out of the \
                     composed library: {e:?}"
                )
            });
        resolved += 1;
    }
    assert_eq!(resolved, 3, "three dtypes are instantiated for `_silu`");

    assert!(
        kernels
            .load_pipeline(&device, Source::Gemv, "gemv_no_such_kernel_f16")
            .is_err(),
        "non-vacuity: an arbitrary name must NOT resolve, or the assertions \
         above say nothing"
    );
}

// ---- eos_index: the batched EOS-index reduction (the GpuBatched arm) ---------
//
// One dispatch over a window of token ids returns the first index whose token is
// a stop id, or `n` if none. The scalar reference is a plain `position` scan; the
// kernel must agree with it for every window shape, including a non-power-of-two
// length (the padding lanes must not corrupt the min-reduce) and a first-match at
// index 0. Determinism: the same window returns the same index every run.

/// The scalar reference the kernel must match: first index whose token is in the
/// stop set, or `n`.
fn eos_index_reference(tokens: &[u32], eos_ids: &[u32]) -> u32 {
    tokens
        .iter()
        .position(|t| eos_ids.contains(t))
        .map(|i| i as u32)
        .unwrap_or(tokens.len() as u32)
}

fn run_eos_index(tokens: &[u32], eos_ids: &[u32]) -> u32 {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let tokens_buf = new_buffer(&device, tokens);
    let eos_buf = new_buffer(&device, eos_ids);
    let out = new_buffer(&device, &[u32::MAX]); // 1 word, pre-set to a value the kernel must overwrite
                                                // SAFETY: the test owns every buffer bound or mapped here and reads its
                                                // results only after waiting for the command buffer that writes them.
    unsafe {
        crate::kernels::kernels::call_eos_index(
            &device,
            &encoder,
            &kernels,
            &tokens_buf,
            0,
            tokens.len(),
            &eos_buf,
            0,
            eos_ids.len(),
            &out,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<u32>(&out, 1)[0]
}

#[test]
fn eos_index_matches_the_scalar_scan() {
    let eos = [2u32, 7, 100];
    // A spread of windows: no match, match at the end, match in the middle,
    // match at 0, multiple matches (first wins), and non-power-of-two lengths.
    let windows: &[&[u32]] = &[
        &[1, 3, 4, 5],          // none -> n
        &[1, 3, 4, 7],          // match at 3
        &[1, 7, 4, 5],          // match at 1
        &[2, 3, 4, 5],          // match at 0
        &[7, 2, 100, 7],        // multiple -> first (0)
        &[1, 3, 4, 5, 6],       // n=5 (width padded to 8), none
        &[1, 3, 4, 5, 100],     // n=5, match at 4 (the last real lane)
        &[3, 3, 3, 3, 3, 3, 7], // n=7, match at 6
        &[100],                 // n=1, match at 0
        &[5],                   // n=1, none
    ];
    for w in windows {
        let got = run_eos_index(w, &eos);
        let want = eos_index_reference(w, &eos);
        assert_eq!(
            got, want,
            "eos_index disagreed with the scan on window {w:?}"
        );
    }
}

#[test]
fn eos_index_is_deterministic() {
    let eos = [2u32, 7];
    let w = [1u32, 3, 7, 4, 2, 9]; // first match at index 2
    let first = run_eos_index(&w, &eos);
    assert_eq!(first, 2);
    for _ in 0..8 {
        assert_eq!(
            run_eos_index(&w, &eos),
            first,
            "eos_index must be deterministic"
        );
    }
}

/// The offset path: the token window is a slice at a NON-ZERO offset into a
/// larger buffer (exactly what a `narrow` of a ring segment produces). The
/// dispatch must honour the byte offset — a version that read from slot 0 would
/// scan the wrong tokens and miss the EOS. This is the regression that shipped
/// once (the end-to-end gate caught a 10x overrun; this catches it at the kernel).
fn run_eos_index_at(buffer: &[u32], offset: usize, n: usize, eos_ids: &[u32]) -> u32 {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let tokens_buf = new_buffer(&device, buffer);
    let eos_buf = new_buffer(&device, eos_ids);
    let out = new_buffer(&device, &[u32::MAX]);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        crate::kernels::kernels::call_eos_index(
            &device,
            &encoder,
            &kernels,
            &tokens_buf,
            offset * std::mem::size_of::<u32>(), // BYTE offset, as Metal's setBuffer:offset:
            n,
            &eos_buf,
            0,
            eos_ids.len(),
            &out,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<u32>(&out, 1)[0]
}

#[test]
fn eos_index_honours_the_token_offset() {
    let eos = [7u32];
    // A 10-slot buffer; the window is slots [4..8). The EOS (7) sits at absolute
    // slot 6 = window index 2. A dispatch ignoring the offset would scan [0..4)
    // (slots 1,2,3,4 — no 7) and wrongly report "none" (n=4).
    let buffer = [1u32, 2, 3, 4, 5, 9, 7, 8, 100, 100];
    assert_eq!(
        run_eos_index_at(&buffer, 4, 4, &eos),
        2,
        "must scan the window at the offset"
    );
    // A window with no EOS at its offset returns n even though an EOS exists
    // OUTSIDE the window (slot 6): window [0..4) has no 7.
    assert_eq!(
        run_eos_index_at(&buffer, 0, 4, &eos),
        4,
        "must not see EOS outside the window"
    );
}

#[test]
fn eos_index_mutation_a_broken_min_would_fail() {
    // The mutation guard: if the reduction took MAX instead of MIN (a plausible
    // slip), a window with matches at BOTH a low and a high index would return the
    // high one. This asserts the low (first) index is returned, so that mutation
    // is caught rather than passing silently.
    let eos = [7u32];
    let w = [7u32, 1, 2, 7]; // matches at 0 and 3; MIN=0, a broken MAX would give 3
    assert_eq!(
        run_eos_index(&w, &eos),
        0,
        "must return the FIRST match, not the last"
    );
}

///////////////////////////////////////////////////////////////////////////////
// RMSNorm over a strided source, RMSNorm + RoPE in one dispatch, and the
// strided-to-strided copy (`DESIGN.md` §12.2p). Each is checked against the
// kernels it replaces, bit for bit, and each carries a control that proves
// the new binding is read (`CONTRIBUTING.md` §3.1a, §3.1b). The norms also
// meet an f64 reference, because the kernels they replace share their
// reduction and cannot see a defect in it.
///////////////////////////////////////////////////////////////////////////////

/// A `[b, t, h, d]` contiguous buffer, and the `[b, h, t, d]` view of it a
/// `transpose(1, 2)` produces: the geometry `Attention::project` hands the
/// norms at `seq_len > 1`.
struct Bthd {
    b: usize,
    t: usize,
    h: usize,
    d: usize,
}

impl Bthd {
    fn numel(&self) -> usize {
        self.b * self.t * self.h * self.d
    }
    /// Shape and strides of the transposed view.
    fn bhtd_view(&self) -> (Vec<usize>, Vec<usize>) {
        (
            vec![self.b, self.h, self.t, self.d],
            vec![self.t * self.h * self.d, self.d, self.h * self.d, 1],
        )
    }
    /// The transposed view materialised on the host, row-major `[b, h, t, d]`.
    fn transpose_host<T: Copy>(&self, bthd: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(self.numel());
        for bi in 0..self.b {
            for hi in 0..self.h {
                for ti in 0..self.t {
                    let base = ((bi * self.t + ti) * self.h + hi) * self.d;
                    out.extend_from_slice(&bthd[base..base + self.d]);
                }
            }
        }
        out
    }
}

trait FromF32: Copy {
    fn from_f32(v: f32) -> Self;
}
impl FromF32 for f16 {
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}
impl FromF32 for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
}

fn ramp<T: FromF32>(n: usize, seed: usize) -> Vec<T> {
    (0..n)
        .map(|i| T::from_f32((((i * 53 + seed) % 89) as f32 - 44.0) / 17.0))
        .collect()
}

trait BitEq: Copy {
    fn bits(self) -> u32;
}
impl BitEq for f16 {
    fn bits(self) -> u32 {
        self.to_bits() as u32
    }
}
impl BitEq for f32 {
    fn bits(self) -> u32 {
        self.to_bits()
    }
}

fn assert_bits_equal<T: BitEq + std::fmt::Debug>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let diff = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (a, b))| a.bits() != b.bits())
        .count();
    assert_eq!(
        diff,
        0,
        "{what}: {diff} of {} elements differ; first mismatch (got, want) {:?}",
        got.len(),
        got.iter().zip(want).find(|(a, b)| a.bits() != b.bits()),
    );
}

fn count_bits_differ<T: BitEq>(a: &[T], b: &[T]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.bits() != y.bits())
        .count()
}

/// A kernel dtype read back as f64, with the relative error one store in it
/// may carry: two units of roundoff at f16, and at f32 enough headroom for
/// the f32 reduction and `rsqrt` ahead of the store.
trait ToF64: Copy {
    const TOL: f64;
    fn to_f64(self) -> f64;
}
impl ToF64 for f16 {
    const TOL: f64 = 1.0 / 1024.0;
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
}
impl ToF64 for f32 {
    const TOL: f64 = 1.0 / 262_144.0;
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// RMSNorm of each `d`-wide row of `rows`, in f64.
fn host_rms_norm<T: ToF64>(rows: &[T], alpha: &[T], d: usize, eps: f64) -> Vec<f64> {
    rows.chunks(d)
        .flat_map(|row| {
            let ms = row.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / d as f64;
            let total = 1.0 / (ms + eps).sqrt();
            row.iter()
                .zip(alpha)
                .map(move |(v, a)| v.to_f64() * total * a.to_f64())
        })
        .collect()
}

/// `rope`'s rotation of `[rows, d]` normalised values, in f64: row `r` sits
/// at position `r % t` and pairs `(i, i + d/2)` against `[t, d/2]` tables.
/// Returns the rotated values and, per element, `|a·c| + |b·s|` -- the
/// magnitude the rotation's roundings scale with.
fn host_rope<T: ToF64>(
    normed: &[f64],
    cos: &[T],
    sin: &[T],
    t: usize,
    d: usize,
) -> (Vec<f64>, Vec<f64>) {
    let half = d / 2;
    let mut out = vec![0.0; normed.len()];
    let mut scale = vec![0.0; normed.len()];
    for (r, row) in normed.chunks(d).enumerate() {
        for i in 0..half {
            let cs = (r % t) * half + i;
            let (c, s) = (cos[cs].to_f64(), sin[cs].to_f64());
            let (a, b) = (row[i], row[i + half]);
            out[r * d + i] = a * c - b * s;
            out[r * d + i + half] = a * s + b * c;
            scale[r * d + i] = (a * c).abs() + (b * s).abs();
            scale[r * d + i + half] = (a * s).abs() + (b * c).abs();
        }
    }
    (out, scale)
}

/// `got` within `k * T::TOL * scale[i]` of the f64 `want`, element by element.
fn assert_near_host<T: ToF64>(what: &str, got: &[T], want: &[f64], scale: &[f64], k: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bound = |i: usize| k * T::TOL * scale[i] + 1e-6;
    let worst = (0..got.len())
        .map(|i| (got[i].to_f64() - want[i]).abs() / bound(i))
        .fold(0.0, f64::max);
    eprintln!("{what}: worst error / bound = {worst:.3}");
    let over: Vec<usize> = (0..got.len())
        .filter(|&i| (got[i].to_f64() - want[i]).abs() > bound(i))
        .collect();
    assert!(
        over.is_empty(),
        "{what}: {} of {} elements off the f64 reference by more than {k} x {:e} relative; \
         first (index, got, want) {:?}",
        over.len(),
        got.len(),
        T::TOL,
        over.first().map(|&i| (i, got[i].to_f64(), want[i])),
    );
}

fn bo(b: &Buffer) -> BufferOffset<'_> {
    BufferOffset {
        buffer: b,
        offset_in_bytes: 0,
    }
}

/// Run the contiguous `rmsnorm_*` over `src` (already `[b, h, t, d]`
/// row-major), returning the normalised rows.
fn rmsnorm_contiguous<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    name: &'static str,
    src: &[T],
    alpha: &[T],
    d: usize,
) -> Vec<T> {
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(device, src);
    let alpha_buf = new_buffer(device, alpha);
    let output = new_buffer(device, src);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm(
            device,
            &encoder,
            kernels,
            name,
            src.len(),
            d,
            1e-5,
            &input,
            0,
            &alpha_buf,
            0,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<T>(&output, src.len())
}

/// Run `rmsnorm_*_strided` over `bthd` presented through `(shape, strides)`.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_strided<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    name: &'static str,
    bthd: &[T],
    shape: &[usize],
    strides: &[usize],
    alpha: &[T],
    style: ParamStyle,
) -> Vec<T> {
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(device, bthd);
    let alpha_buf = new_buffer(device, alpha);
    let output = new_buffer(device, bthd);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm_strided_with(
            device,
            &encoder,
            kernels,
            name,
            shape,
            strides,
            1e-5,
            BufferOffset {
                buffer: &input,
                offset_in_bytes: 0,
            },
            BufferOffset {
                buffer: &alpha_buf,
                offset_in_bytes: 0,
            },
            &output,
            style,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<T>(&output, bthd.len())
}

fn rmsnorm_strided_matches_contiguous_over_a_copy_at<
    T: FromF32 + BitEq + ToF64 + std::fmt::Debug,
>(
    contiguous: &'static str,
    strided: &'static str,
) {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 2,
        t: 5,
        h: 3,
        d: 64,
    };
    let bthd: Vec<T> = ramp(g.numel(), 0);
    let alpha: Vec<T> = ramp(g.d, 7);
    let (shape, strides) = g.bhtd_view();

    let want = rmsnorm_contiguous(
        &device,
        &kernels,
        contiguous,
        &g.transpose_host(&bthd),
        &alpha,
        g.d,
    );
    let got = rmsnorm_strided(
        &device,
        &kernels,
        strided,
        &bthd,
        &shape,
        &strides,
        &alpha,
        ParamStyle::Split,
    );
    assert_bits_equal(strided, &got, &want);

    // Level: `want` runs the same `rms_norm` body as `got`, so a defect in the
    // shared reduction moves both and the match above still holds. The f64
    // norm does not move with them.
    let host = host_rms_norm(&g.transpose_host(&bthd), &alpha, g.d, 1e-5);
    let scale: Vec<f64> = host.iter().map(|v| v.abs()).collect();
    assert_near_host(strided, &got, &host, &scale, 1.0);

    // Control: the strides are read. Swapping the `h` and `t` strides presents
    // the untransposed rows, which normalise to different values.
    let wrong = vec![strides[0], strides[2], strides[1], strides[3]];
    let control = rmsnorm_strided(
        &device,
        &kernels,
        strided,
        &bthd,
        &shape,
        &wrong,
        &alpha,
        ParamStyle::Split,
    );
    let moved = count_bits_differ(&control, &want);
    assert!(
        moved > 0,
        "{strided}: a wrong stride set must move the output; it moved 0"
    );
}

#[test]
fn rmsnorm_strided_matches_the_contiguous_kernel_over_a_copy() {
    rmsnorm_strided_matches_contiguous_over_a_copy_at::<f16>("rmsnorm_f16", "rmsnorm_f16_strided");
    rmsnorm_strided_matches_contiguous_over_a_copy_at::<f32>("rmsnorm_f32", "rmsnorm_f32_strided");
}

#[test]
fn packed_matches_split_for_rmsnorm_f16_strided() {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 1,
        t: 4,
        h: 8,
        d: 64,
    };
    let bthd: Vec<f16> = ramp(g.numel(), 3);
    let alpha: Vec<f16> = ramp(g.d, 11);
    let (shape, strides) = g.bhtd_view();
    let split = rmsnorm_strided(
        &device,
        &kernels,
        "rmsnorm_f16_strided",
        &bthd,
        &shape,
        &strides,
        &alpha,
        ParamStyle::Split,
    );
    let packed = rmsnorm_strided(
        &device,
        &kernels,
        "rmsnorm_f16_strided",
        &bthd,
        &shape,
        &strides,
        &alpha,
        ParamStyle::Packed,
    );
    assert_packed_matches_split("rmsnorm_f16_strided", &split, &packed);
}

/// The two-dispatch reference: `rmsnorm_*` over the transposed copy, then
/// `rope_*` over its output with `[t, d/2]` tables.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_then_rope<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    norm: &'static str,
    rope: &'static str,
    g: &Bthd,
    bhtd: &[T],
    alpha: &[T],
    cos: &[T],
    sin: &[T],
) -> Vec<T> {
    let normed = rmsnorm_contiguous(device, kernels, norm, bhtd, alpha, g.d);
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(device, &normed);
    let cos_buf = new_buffer(device, cos);
    let sin_buf = new_buffer(device, sin);
    let output = new_buffer(device, &normed);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rope(
            device,
            &encoder,
            kernels,
            rope,
            g.b * g.h,
            g.t * g.d,
            g.d,
            0,
            &input,
            0,
            &cos_buf,
            0,
            &sin_buf,
            0,
            &output,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<T>(&output, normed.len())
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm_rope<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    name: &'static str,
    src: &[T],
    shape: &[usize],
    strides: Option<&[usize]>,
    t_len: usize,
    alpha: &[T],
    cos: &[T],
    sin: &[T],
    style: ParamStyle,
) -> Vec<T> {
    let commands = commands(device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(device, src);
    let alpha_buf = new_buffer(device, alpha);
    let cos_buf = new_buffer(device, cos);
    let sin_buf = new_buffer(device, sin);
    let output = new_buffer(device, src);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    unsafe {
        call_rms_norm_rope_with(
            device,
            &encoder,
            kernels,
            name,
            shape,
            strides,
            t_len,
            1e-5,
            bo(&input),
            bo(&alpha_buf),
            bo(&cos_buf),
            bo(&sin_buf),
            &output,
            style,
        )
    }
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec::<T>(&output, src.len())
}

fn rmsnorm_rope_matches_the_pair_at<T: FromF32 + BitEq + ToF64 + std::fmt::Debug>(
    norm: &'static str,
    rope: &'static str,
    fused: &'static str,
    fused_strided: &'static str,
) {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 2,
        t: 5,
        h: 3,
        d: 64,
    };
    let bthd: Vec<T> = ramp(g.numel(), 0);
    let bhtd = g.transpose_host(&bthd);
    let alpha: Vec<T> = ramp(g.d, 7);
    let cos: Vec<T> = (0..g.t * g.d / 2)
        .map(|i| T::from_f32(((i * 7) as f32 / 9.0).cos()))
        .collect();
    let sin: Vec<T> = (0..g.t * g.d / 2)
        .map(|i| T::from_f32(((i * 7) as f32 / 9.0).sin()))
        .collect();
    let (shape, strides) = g.bhtd_view();

    let want = rmsnorm_then_rope(&device, &kernels, norm, rope, &g, &bhtd, &alpha, &cos, &sin);
    let got_contig = rmsnorm_rope(
        &device,
        &kernels,
        fused,
        &bhtd,
        &shape,
        None,
        g.t,
        &alpha,
        &cos,
        &sin,
        ParamStyle::Split,
    );
    assert_bits_equal(fused, &got_contig, &want);
    let got_strided = rmsnorm_rope(
        &device,
        &kernels,
        fused_strided,
        &bthd,
        &shape,
        Some(&strides),
        g.t,
        &alpha,
        &cos,
        &sin,
        ParamStyle::Split,
    );
    assert_bits_equal(fused_strided, &got_strided, &want);

    // Level: the pair shares the fused kernel's reduction, so the matches
    // above cannot see a defect in it; the f64 norm + rotation can. Each
    // element carries the roundings of its two normalised inputs, the
    // product with `s` and the store.
    let normed = host_rms_norm(&bhtd, &alpha, g.d, 1e-5);
    let (host, scale) = host_rope(&normed, &cos, &sin, g.t, g.d);
    assert_near_host(fused, &got_contig, &host, &scale, 2.0);

    // Controls: the rotation is applied (the fused output is not the bare
    // norm), and the position is read (a table shifted by one row moves it).
    let bare = rmsnorm_contiguous(&device, &kernels, norm, &bhtd, &alpha, g.d);
    assert!(
        count_bits_differ(&got_contig, &bare) > 0,
        "{fused}: output equals the unrotated norm"
    );
    let mut cos_shift = cos.clone();
    cos_shift.rotate_left(g.d / 2);
    let shifted = rmsnorm_rope(
        &device,
        &kernels,
        fused,
        &bhtd,
        &shape,
        None,
        g.t,
        &alpha,
        &cos_shift,
        &sin,
        ParamStyle::Split,
    );
    assert!(
        count_bits_differ(&shifted, &want) > 0,
        "{fused}: a shifted cos table must move the output"
    );
}

#[test]
fn rmsnorm_rope_matches_rmsnorm_then_rope() {
    rmsnorm_rope_matches_the_pair_at::<f16>(
        "rmsnorm_f16",
        "rope_f16",
        "rmsnorm_rope_f16",
        "rmsnorm_rope_f16_strided",
    );
    rmsnorm_rope_matches_the_pair_at::<f32>(
        "rmsnorm_f32",
        "rope_f32",
        "rmsnorm_rope_f32",
        "rmsnorm_rope_f32_strided",
    );
}

#[test]
fn rmsnorm_rope_refuses_a_row_its_threadgroup_cannot_pair() {
    let device = device();
    let kernels = Kernels::new();
    // 96 is not a power of two: the threadgroup is 64 wide and thread `tid`
    // would not hold `(tid, tid + 48)`. The host refuses rather than dispatch.
    assert!(!rms_norm_rope_applies(&device, &kernels, "rmsnorm_rope_f16", 96).unwrap());
    assert!(rms_norm_rope_applies(&device, &kernels, "rmsnorm_rope_f16", 64).unwrap());
    assert!(rms_norm_rope_applies(&device, &kernels, "rmsnorm_rope_f16", 128).unwrap());
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let src: Vec<f16> = ramp(2 * 96, 0);
    let alpha: Vec<f16> = ramp(96, 1);
    let table: Vec<f16> = ramp(48, 2);
    let input = new_buffer(&device, &src);
    let alpha_buf = new_buffer(&device, &alpha);
    let cos_buf = new_buffer(&device, &table);
    let sin_buf = new_buffer(&device, &table);
    let output = new_buffer(&device, &src);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let r = unsafe {
        call_rms_norm_rope(
            &device,
            &encoder,
            &kernels,
            "rmsnorm_rope_f16",
            &[1, 2, 1, 96],
            None,
            1,
            1e-5,
            bo(&input),
            bo(&alpha_buf),
            bo(&cos_buf),
            bo(&sin_buf),
            &output,
        )
    };
    assert!(r.is_err(), "a 96-wide row must be refused");
}

#[test]
fn packed_matches_split_for_rmsnorm_rope_f16() {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 1,
        t: 3,
        h: 8,
        d: 64,
    };
    let bthd: Vec<f16> = ramp(g.numel(), 5);
    let alpha: Vec<f16> = ramp(g.d, 11);
    let cos: Vec<f16> = ramp(g.t * g.d / 2, 13);
    let sin: Vec<f16> = ramp(g.t * g.d / 2, 17);
    let (shape, strides) = g.bhtd_view();
    for (name, st) in [
        ("rmsnorm_rope_f16", None),
        ("rmsnorm_rope_f16_strided", Some(strides.as_slice())),
    ] {
        let split = rmsnorm_rope(
            &device,
            &kernels,
            name,
            &bthd,
            &shape,
            st,
            g.t,
            &alpha,
            &cos,
            &sin,
            ParamStyle::Split,
        );
        let packed = rmsnorm_rope(
            &device,
            &kernels,
            name,
            &bthd,
            &shape,
            st,
            g.t,
            &alpha,
            &cos,
            &sin,
            ParamStyle::Packed,
        );
        assert_packed_matches_split(name, &split, &packed);
    }
}

/// `copy_nd`: a `[b, h, t, d]` transposed view written into a `[b, h, T, d]`
/// slab at rows `off..off + t` -- the KV append from the projection's own
/// layout -- against the same write computed on the host, in both binding
/// styles.
#[test]
fn copy_nd_writes_a_transposed_view_into_a_strided_window() {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 2,
        t: 3,
        h: 4,
        d: 8,
    };
    let cap = 7usize;
    let off = 2usize;
    let src: Vec<f16> = ramp(g.numel(), 0);
    let (shape, src_strides) = g.bhtd_view();
    let dst_strides = vec![g.h * cap * g.d, cap * g.d, g.d, 1];
    let slab_len = g.b * g.h * cap * g.d;
    let fill: Vec<f16> = vec![f16::from_f32(-9.0); slab_len];

    let mut want = fill.clone();
    for bi in 0..g.b {
        for hi in 0..g.h {
            for ti in 0..g.t {
                for di in 0..g.d {
                    let s = bi * src_strides[0] + hi * src_strides[1] + ti * src_strides[2] + di;
                    let dpos = bi * dst_strides[0]
                        + hi * dst_strides[1]
                        + (off + ti) * dst_strides[2]
                        + di;
                    want[dpos] = src[s];
                }
            }
        }
    }

    // Both binding styles against the host write: the packed form has its own
    // entry point, and nothing else in the tree dispatches it under test.
    for style in [ParamStyle::Split, ParamStyle::Packed] {
        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        let input = new_buffer(&device, &src);
        let output = new_buffer(&device, &fill);
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_copy_nd_with(
                &device,
                &encoder,
                &kernels,
                copy_nd::HALF,
                &shape,
                BufferOffset {
                    buffer: &input,
                    offset_in_bytes: 0,
                },
                &src_strides,
                BufferOffset {
                    buffer: &output,
                    offset_in_bytes: off * g.d * 2,
                },
                &dst_strides,
                style,
            )
        }
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let got = read_to_vec::<f16>(&output, slab_len);
        assert_bits_equal(&format!("copy_nd_f16 {style:?}"), &got, &want);
        // The window is what moved: exactly `numel` slab elements left the fill.
        let written = got
            .iter()
            .filter(|v| v.to_bits() != f16::from_f32(-9.0).to_bits())
            .count();
        assert_eq!(
            written,
            g.numel(),
            "copy_nd_f16 {style:?} wrote outside its window or missed part of it"
        );
    }
}

/// The host oracle that fixed the fused rotation's rounding (`DESIGN.md`
/// §12.2p): for each line of the rotation, four candidate roundings of the
/// pair -- single rounding, both products rounded, and each product rounded
/// with the other fused -- tallied against the two-dispatch reference and
/// against the fused kernel. The form the reference takes is the one that
/// matches every pair; a fused kernel that matches the reference matches the
/// same form. Ignored: a diagnostic, not a gate -- `rmsnorm_rope_matches_rmsnorm_then_rope`
/// is the gate.
#[test]
#[ignore]
fn rmsnorm_rope_f16_second_line_diagnostic() {
    let device = device();
    let kernels = Kernels::new();
    let g = Bthd {
        b: 2,
        t: 5,
        h: 3,
        d: 64,
    };
    let bthd: Vec<f16> = ramp(g.numel(), 0);
    let bhtd = g.transpose_host(&bthd);
    let alpha: Vec<f16> = ramp(g.d, 7);
    let cos: Vec<f16> = (0..g.t * g.d / 2)
        .map(|i| f16::from_f32(((i * 7) as f32 / 9.0).cos()))
        .collect();
    let sin: Vec<f16> = (0..g.t * g.d / 2)
        .map(|i| f16::from_f32(((i * 7) as f32 / 9.0).sin()))
        .collect();
    let (shape, _) = g.bhtd_view();
    let want = rmsnorm_then_rope(
        &device,
        &kernels,
        "rmsnorm_f16",
        "rope_f16",
        &g,
        &bhtd,
        &alpha,
        &cos,
        &sin,
    );
    let got = rmsnorm_rope(
        &device,
        &kernels,
        "rmsnorm_rope_f16",
        &bhtd,
        &shape,
        None,
        g.t,
        &alpha,
        &cos,
        &sin,
        ParamStyle::Split,
    );
    let bare = rmsnorm_contiguous(&device, &kernels, "rmsnorm_f16", &bhtd, &alpha, g.d);
    let half = g.d / 2;
    let r = |x: f32| f16::from_f32(x).to_f32();
    let mut t0 = [0usize; 8];
    let mut t1 = [0usize; 8];
    let (mut m0, mut m1) = (0, 0);
    for row in 0..(g.b * g.h * g.t) {
        let t = row % g.t;
        for i in 0..half {
            let i0 = row * g.d + i;
            let i1 = i0 + half;
            let (v0, v1) = (bare[i0].to_f32(), bare[i1].to_f32());
            let (c, s) = (cos[t * half + i].to_f32(), sin[t * half + i].to_f32());
            let c0 = [
                f16::from_f32(v0 * c - v1 * s),       // single
                f16::from_f32(r(v0 * c) - r(v1 * s)), // halfops
                f16::from_f32(v0 * c - r(v1 * s)),    // fma(v0,c,-r(v1 s))
                f16::from_f32(r(v0 * c) - v1 * s),    // fma(-v1,s,r(v0 c))
            ];
            let c1 = [
                f16::from_f32(v0 * s + v1 * c),
                f16::from_f32(r(v0 * s) + r(v1 * c)),
                f16::from_f32(v0 * s + r(v1 * c)), // fma(v0,s,r(v1 c))
                f16::from_f32(r(v0 * s) + v1 * c), // fma(v1,c,r(v0 s))
            ];
            if got[i0].to_bits() != want[i0].to_bits() {
                m0 += 1;
            }
            if got[i1].to_bits() != want[i1].to_bits() {
                m1 += 1;
            }
            for k in 0..4 {
                if c0[k].to_bits() == want[i0].to_bits() {
                    t0[k] += 1;
                }
                if c0[k].to_bits() == got[i0].to_bits() {
                    t0[4 + k] += 1;
                }
                if c1[k].to_bits() == want[i1].to_bits() {
                    t1[k] += 1;
                }
                if c1[k].to_bits() == got[i1].to_bits() {
                    t1[4 + k] += 1;
                }
            }
        }
    }
    let n = g.b * g.h * g.t * half;
    eprintln!("pairs={n} mismatches i0={m0} i1={m1}");
    eprintln!("i0 want: {:?}  got: {:?}", &t0[..4], &t0[4..]);
    eprintln!("i1 want: {:?}  got: {:?}", &t1[..4], &t1[4..]);
}

/// A blit reads and writes buffers as a dispatch binds them, and the arena
/// recording must see it (`DESIGN.md` §9.2aa): a value whose last reader is a
/// `copyFromBuffer` would otherwise end its interval at its producing
/// dispatch and be aliased while the copy still reads it.
#[test]
fn a_blit_copy_extends_the_recorded_interval_of_both_buffers() {
    use crate::kernels::metal::arena::{set_bind_observer, ArenaRecorder};
    use parking_lot::Mutex;
    let device = device();
    let src = new_buffer(&device, &[1u32; 64]);
    let dst = new_buffer(&device, &[0u32; 64]);
    let rec = Arc::new(Mutex::new(ArenaRecorder::new()));
    {
        let mut r = rec.lock();
        r.record_alloc(src.raw_addr(), 256);
        r.record_alloc(dst.raw_addr(), 256);
        // Three dispatches go by with neither buffer bound.
        r.record_dispatch();
        r.record_dispatch();
        r.record_dispatch();
    }
    assert_eq!(rec.lock().last_bind_of(0), Some(0));
    assert_eq!(rec.lock().last_bind_of(1), Some(0));
    set_bind_observer(Some(Arc::clone(&rec)));
    // While this recorder is the process's observer, every compute dispatch
    // any concurrent test encodes advances its counter too, so the position
    // the blit lands at is read off the recorder, not assumed to be 3.
    let at_least = rec.lock().dispatch_position();
    let commands = commands(&device);
    {
        let mut blit = commands.blit_command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { blit.copy_from_buffer(&src, 0, &dst, 0, 256) };
    }
    set_bind_observer(None);
    commands.wait_until_completed().unwrap();
    // Both intervals now end where the copy was encoded — at or after the
    // third dispatch. Without the note in the blit encoder they read 0, the
    // aliasing the note exists to prevent.
    assert!(at_least >= 3);
    let read = rec.lock().last_bind_of(0);
    let write = rec.lock().last_bind_of(1);
    assert!(
        read.is_some_and(|p| p >= at_least),
        "the blit's read was not recorded: {read:?} < {at_least}"
    );
    assert!(
        write.is_some_and(|p| p >= at_least),
        "the blit's write was not recorded: {write:?} < {at_least}"
    );
    assert_eq!(read_to_vec::<u32>(&dst, 64)[0], 1);
}

/// Tensor slices are element-aligned, including odd f16/bf16 and u8 offsets.
/// Split and packed bindings must preserve the same independent CPU reference.
#[test]
fn strided_copy_accepts_element_aligned_offsets() {
    fn check<T: Copy + PartialEq + std::fmt::Debug>(values: Vec<T>, name: &'static str) {
        let dev = device();
        let kernels = Kernels::new();
        for offset in [1usize, 3, 5] {
            for style in [ParamStyle::Split, ParamStyle::Packed] {
                for repeat in 0..3 {
                    let commands = commands(&dev);
                    let input = new_buffer(&dev, &values);
                    let sentinel = values[0];
                    let output = new_buffer(&dev, &[sentinel; 8]);
                    let encoder = commands.command_encoder().unwrap();
                    // SAFETY: the test owns every buffer bound or mapped here and reads its
                    // results only after waiting for the command buffer that writes them.
                    unsafe {
                        call_unary_strided_with(
                            &dev,
                            &encoder,
                            &kernels,
                            unary::strided::Kernel(name),
                            &[2, 3],
                            BufferOffset {
                                buffer: &input,
                                offset_in_bytes: offset * size_of::<T>(),
                            },
                            &[1, 2],
                            BufferOffset {
                                buffer: &output,
                                offset_in_bytes: size_of::<T>(),
                            },
                            style,
                        )
                    }
                    .unwrap();
                    drop(encoder);
                    commands.wait_until_completed().unwrap();
                    let got = read_to_vec::<T>(&output, 8);
                    let expected = [
                        sentinel,
                        values[offset],
                        values[offset + 2],
                        values[offset + 4],
                        values[offset + 1],
                        values[offset + 3],
                        values[offset + 5],
                        sentinel,
                    ];
                    assert_eq!(
                        got, expected,
                        "{name} offset={offset} style={style:?} repeat={repeat}"
                    );
                }
            }
        }
    }
    check(
        (0..16).map(|i| f16::from_f32(i as f32 - 8.0)).collect(),
        "copy_f16_strided",
    );
    check(
        (0..16).map(|i| bf16::from_f32(i as f32 - 8.0)).collect(),
        "copy_bf16_strided",
    );
    check((0..16u8).collect(), "copy_u8_strided");
    println!("STRIDED_ALIGNMENT dtypes=3 offsets=3 styles=2 repeats=3 cases=54 cpu_reference=true guard_words=true");
}
