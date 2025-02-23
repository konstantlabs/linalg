use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linalg::matrix::mat::Matrix;
use linalg::num::{Complex32, Complex64};

fn add_2x2_matrices_i32(c: &mut Criterion) {
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);

    c.bench_function("add 2x2 matrices i32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_2x2_matrices_f32(c: &mut Criterion) {
    let m1: Matrix<f32> = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2: Matrix<f32> = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

    c.bench_function("add 2x2 matrices f32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_2x2_matrices_f64(c: &mut Criterion) {
    let m1: Matrix<f64> = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2: Matrix<f64> = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

    c.bench_function("add 2x2 matrices f64", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_3x3_matrices_f32(c: &mut Criterion) {
    let m1: Matrix<f32> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m2: Matrix<f32> = Matrix::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    c.bench_function("add 3x3 matrices f32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_3x3_matrices_f64(c: &mut Criterion) {
    let m1: Matrix<f64> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m2: Matrix<f64> = Matrix::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    c.bench_function("add 3x3 matrices f64", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_3x3_matrices_i32(c: &mut Criterion) {
    let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let m2: Matrix<i32> = Matrix::new([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);

    c.bench_function("add 3x3 matrices i32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

// fn add_3x3_matrices_i64(c: &mut Criterion) {
//     let m1: Matrix<i64> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//     let m2: Matrix<i64> = Matrix::new([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);

//     c.bench_function("add 3x3 matrices i64", |b| {
//         b.iter(|| black_box(&m1) + black_box(&m2))
//     });
// }

fn add_1024x1024_matrices_f32(c: &mut Criterion) {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    let data1: Vec<f32> = (0..1024 * 1024)
        .map(|_| rng.gen_range(-100.0..100.0))
        .collect();
    let data2: Vec<f32> = (0..1024 * 1024)
        .map(|_| rng.gen_range(-100.0..100.0))
        .collect();

    let m1: Matrix<f32> = Matrix::from_vec(1024, 1024, data1);
    let m2: Matrix<f32> = Matrix::from_vec(1024, 1024, data2);

    c.bench_function("add 1024x1024 matrices f32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_1024x1024_matrices_f64(c: &mut Criterion) {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    let data1: Vec<f64> = (0..1024 * 1024)
        .map(|_| rng.gen_range(-100.0..100.0))
        .collect();
    let data2: Vec<f64> = (0..1024 * 1024)
        .map(|_| rng.gen_range(-100.0..100.0))
        .collect();

    let m1: Matrix<f64> = Matrix::from_vec(1024, 1024, data1);
    let m2: Matrix<f64> = Matrix::from_vec(1024, 1024, data2);

    c.bench_function("add 1024x1024 matrices f64", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn add_1024x1024_matrices_i32(c: &mut Criterion) {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    let data1: Vec<i32> = (0..1024 * 1024).map(|_| rng.gen_range(-100..100)).collect();
    let data2: Vec<i32> = (0..1024 * 1024).map(|_| rng.gen_range(-100..100)).collect();

    let m1: Matrix<i32> = Matrix::from_vec(1024, 1024, data1);
    let m2: Matrix<i32> = Matrix::from_vec(1024, 1024, data2);

    c.bench_function("add 1024x1024 matrices i32", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

fn mul_3x3_matrices_i32(c: &mut Criterion) {
    let m1: Matrix<i32> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    let m2: Matrix<i32> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);

    c.bench_function("mul 3x3 matrices i32", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

fn mul_3x3_matrices_i64(c: &mut Criterion) {
    let m1: Matrix<i64> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    let m2: Matrix<i64> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);

    c.bench_function("mul 3x3 matrices i64", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

fn mul_3x3_matrices_f32(c: &mut Criterion) {
    let m1: Matrix<f32> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    let m2: Matrix<f32> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);

    c.bench_function("mul 3x3 matrices f32", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

fn mul_3x3_matrices_f64(c: &mut Criterion) {
    let m1: Matrix<f64> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    let m2: Matrix<f64> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);

    c.bench_function("mul 3x3 matrices f64", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

// fn mul_512x512_matrices(c: &mut Criterion) {
//     use rand::{thread_rng, Rng};
//     let mut rng = thread_rng();

//     let data1: Vec<f32> = (0..512 * 512)
//         .map(|_| rng.gen_range(-100.0..100.0))
//         .collect();
//     let data2: Vec<f32> = (0..512 * 512)
//         .map(|_| rng.gen_range(-100.0..100.0))
//         .collect();

//     let m1: Matrix<f32> = Matrix::from_vec(512, 512, data1);
//     let m2: Matrix<f32> = Matrix::from_vec(512, 512, data2);

//     c.bench_function("mul 512x512 matrices", |b| {
//         b.iter(|| black_box(&m1) * black_box(&m2))
//     });
// }

fn mul_3x3_complex32_matrices(c: &mut Criterion) {
    let m1: Matrix<Complex32> = Matrix::new([[Complex32::new(2.0, 0.0); 3]; 3]);
    let m2: Matrix<Complex32> = Matrix::new([[Complex32::new(2.0, 0.0); 3]; 3]);

    c.bench_function("mul 3x3 complex32 matrices", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

fn mul_3x3_complex64_matrices(c: &mut Criterion) {
    let m1: Matrix<Complex64> = Matrix::new([[Complex64::new(2.0, 0.0); 3]; 3]);
    let m2: Matrix<Complex64> = Matrix::new([[Complex64::new(2.0, 0.0); 3]; 3]);

    c.bench_function("mul 3x3 complex64 matrices", |b| {
        b.iter(|| black_box(&m1) * black_box(&m2))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = add_2x2_matrices_i32, add_2x2_matrices_f32, add_2x2_matrices_f64, add_3x3_matrices_f32, add_3x3_matrices_f64, add_3x3_matrices_i32,
    mul_3x3_matrices_i32, mul_3x3_matrices_i64, mul_3x3_matrices_f32, mul_3x3_matrices_f64,
    mul_3x3_complex32_matrices, mul_3x3_complex64_matrices,
    add_1024x1024_matrices_f64, add_1024x1024_matrices_f32, add_1024x1024_matrices_i32
}

criterion_main!(benches);
