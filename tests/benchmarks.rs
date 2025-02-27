use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linalg::matrix::mat::Matrix;
use linalg::num::{c32, c64};

pub fn bench_2x2_matrix_adds(c: &mut Criterion) {
    let mut group = c.benchmark_group("2x2 Matrix Addition");

    // i32
    let m1: Matrix<i32> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    let m2: Matrix<i32> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f32
    let m1: Matrix<f32> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    let m2: Matrix<f32> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    let m2: Matrix<f64> = Matrix::new([
        [rand::random(), rand::random()],
        [rand::random(), rand::random()],
    ]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // c32
    let m1: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random(), rand::random()),
            c32::new(rand::random(), rand::random()),
        ],
        [
            c32::new(rand::random(), rand::random()),
            c32::new(rand::random(), rand::random()),
        ],
    ]);
    let m2: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random(), rand::random()),
            c32::new(rand::random(), rand::random()),
        ],
        [
            c32::new(rand::random(), rand::random()),
            c32::new(rand::random(), rand::random()),
        ],
    ]);
    group.bench_function("c32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // c64
    let m1: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random(), rand::random()),
            c64::new(rand::random(), rand::random()),
        ],
        [
            c64::new(rand::random(), rand::random()),
            c64::new(rand::random(), rand::random()),
        ],
    ]);
    let m2: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random(), rand::random()),
            c64::new(rand::random(), rand::random()),
        ],
        [
            c64::new(rand::random(), rand::random()),
            c64::new(rand::random(), rand::random()),
        ],
    ]);
    group.bench_function("c64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    group.finish();
}

pub fn bench_3x3_matrix_adds(c: &mut Criterion) {
    let mut group = c.benchmark_group("3x3 Matrix Addition");

    // i32
    let m1: Matrix<i32> = Matrix::new([
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
    ]);
    let m2: Matrix<i32> = Matrix::new([
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
    ]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f32
    let m1: Matrix<f32> = Matrix::new([
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
    ]);
    let m2: Matrix<f32> = Matrix::new([
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
    ]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
    ]);
    let m2: Matrix<f64> = Matrix::new([
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
    ]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // c32
    let m1: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
    ]);
    let m2: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
    ]);
    group.bench_function("c32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

    // c64
    let m1: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
    ]);
    let m2: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
    ]);
    group.bench_function("c64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));

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

    // c32
    {
        let data1: Vec<c32> = (0..1024 * 1024)
            .map(|_| c32::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
            .collect();
        let data2: Vec<c32> = (0..1024 * 1024)
            .map(|_| c32::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
            .collect();

        let m1: Matrix<c32> = Matrix::from_vec(1024, 1024, data1);
        let m2: Matrix<c32> = Matrix::from_vec(1024, 1024, data2);

        group.bench_function("c32", |b| b.iter(|| black_box(&m1) + black_box(&m2)));
    }

    // c64
    {
        let data1: Vec<c64> = (0..1024 * 1024)
            .map(|_| c64::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
            .collect();
        let data2: Vec<c64> = (0..1024 * 1024)
            .map(|_| c64::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
            .collect();

        let m1: Matrix<c64> = Matrix::from_vec(1024, 1024, data1);
        let m2: Matrix<c64> = Matrix::from_vec(1024, 1024, data2);

        group.bench_function("c64", |b| b.iter(|| black_box(&m1) + black_box(&m2)));
    }

    group.finish();
}

pub fn bench_3x3_matrix_muls(c: &mut Criterion) {
    let mut group = c.benchmark_group("3x3 Matrix Multiplication");

    // i32
    let m1: Matrix<i32> = Matrix::new([
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
    ]);
    let m2: Matrix<i32> = Matrix::new([
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
        [
            rand::random::<i32>(),
            rand::random::<i32>(),
            rand::random::<i32>(),
        ],
    ]);
    group.bench_function("i32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // i64
    let m1: Matrix<i64> = Matrix::new([
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
    ]);
    let m2: Matrix<i64> = Matrix::new([
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
        [
            rand::random::<i64>(),
            rand::random::<i64>(),
            rand::random::<i64>(),
        ],
    ]);
    group.bench_function("i64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // f32
    let m1: Matrix<f32> = Matrix::new([
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
    ]);
    let m2: Matrix<f32> = Matrix::new([
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
        [
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        ],
    ]);
    group.bench_function("f32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // f64
    let m1: Matrix<f64> = Matrix::new([
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
    ]);
    let m2: Matrix<f64> = Matrix::new([
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
        [
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        ],
    ]);
    group.bench_function("f64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // c32
    let m1: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
    ]);
    let m2: Matrix<c32> = Matrix::new([
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
        [
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
            c32::new(rand::random::<f32>(), rand::random::<f32>()),
        ],
    ]);
    group.bench_function("c32", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    // c64
    let m1: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
    ]);
    let m2: Matrix<c64> = Matrix::new([
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
        [
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
            c64::new(rand::random::<f64>(), rand::random::<f64>()),
        ],
    ]);
    group.bench_function("c64", |b| b.iter(|| black_box(&m1) * black_box(&m2)));

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets =
        bench_2x2_matrix_adds,
        bench_3x3_matrix_adds,
        bench_1024x1024_matrix_adds,
        bench_3x3_matrix_muls,
}

criterion_main!(benches);
