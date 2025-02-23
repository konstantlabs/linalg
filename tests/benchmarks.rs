use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linalg::matrix::mat::Matrix;
use linalg::num::{Complex32, Complex64};

pub fn bench_2x2_matrix_adds(c: &mut Criterion) {
    let mut group = c.benchmark_group("2x2 Matrix Addition");

    // i32
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f32
    let m1: Matrix<f32> = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2: Matrix<f32> = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2: Matrix<f64> = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    group.finish();
}

pub fn bench_3x3_matrix_adds(c: &mut Criterion) {
    let mut group = c.benchmark_group("3x3 Matrix Addition");

    // f32
    let m1: Matrix<f32> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m2: Matrix<f32> = Matrix::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m2: Matrix<f64> = Matrix::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // i32
    let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let m2: Matrix<i32> = Matrix::new([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    group.finish();
}

pub fn bench_1024x1024_matrix_adds(c: &mut Criterion) {
    let mut group = c.benchmark_group("1024x1024 Matrix Addition");
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();

    // f32
    {
        let data1: Vec<f32> = (0..1024 * 1024)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();
        let data2: Vec<f32> = (0..1024 * 1024)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();

        let m1: Matrix<f32> = Matrix::from_vec(1024, 1024, data1);
        let m2: Matrix<f32> = Matrix::from_vec(1024, 1024, data2);

        group.bench_function("f32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));
    }

    // f64
    {
        let data1: Vec<f64> = (0..1024 * 1024)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();
        let data2: Vec<f64> = (0..1024 * 1024)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();

        let m1: Matrix<f64> = Matrix::from_vec(1024, 1024, data1);
        let m2: Matrix<f64> = Matrix::from_vec(1024, 1024, data2);

        group.bench_function("f64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));
    }

    // i32
    {
        let data1: Vec<i32> = (0..1024 * 1024).map(|_| rng.gen_range(-100..100)).collect();
        let data2: Vec<i32> = (0..1024 * 1024).map(|_| rng.gen_range(-100..100)).collect();

        let m1: Matrix<i32> = Matrix::from_vec(1024, 1024, data1);
        let m2: Matrix<i32> = Matrix::from_vec(1024, 1024, data2);

        group.bench_function("i32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));
    }

    group.finish();
}

pub fn bench_3x3_matrix_muls(c: &mut Criterion) {
    let mut group = c.benchmark_group("3x3 Matrix Multiplication");

    // i32
    let m1: Matrix<i32> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    let m2: Matrix<i32> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // i64
    let m1: Matrix<i64> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    let m2: Matrix<i64> = Matrix::new([[2, 2, 2], [2, 2, 2], [2, 2, 2]]);
    group.bench_function("i64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // f32
    let m1: Matrix<f32> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    let m2: Matrix<f32> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    let m2: Matrix<f64> = Matrix::new([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // complex32
    let m1: Matrix<Complex32> = Matrix::new([[Complex32::new(2.0, 0.0); 3]; 3]);
    let m2: Matrix<Complex32> = Matrix::new([[Complex32::new(2.0, 0.0); 3]; 3]);
    group.bench_function("complex32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // complex64
    let m1: Matrix<Complex64> = Matrix::new([[Complex64::new(2.0, 0.0); 3]; 3]);
    let m2: Matrix<Complex64> = Matrix::new([[Complex64::new(2.0, 0.0); 3]; 3]);
    group.bench_function("complex64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = bench_2x2_matrix_adds,
    bench_3x3_matrix_adds,
    bench_1024x1024_matrix_adds,
    bench_3x3_matrix_muls,
}

criterion_main!(benches);
