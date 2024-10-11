use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linalg::matrix::mat::Matrix;

fn add_2x2_matrices(c: &mut Criterion) {
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);

    // FIX: cannot add two matrix references
    c.bench_function("add 2x2 matrices", |b| {
        b.iter(|| black_box(&m1) + black_box(&m2))
    });
}

criterion_group!(benches, add_2x2_matrices);
criterion_main!(benches);
