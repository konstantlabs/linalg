use super::{mat::Matrix, simd::SimdOps};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, Sub};

const SIMD_THRESHOLD: usize = 512 * 512; // Minimum elements for SIMD to be worth it

fn add_matrix_impl<'a, T>(m1: &'a Matrix<T>, m2: &'a Matrix<T>) -> Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Sync
        + Send,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    let total_elements = m1.rows * m1.cols;

    if total_elements >= SIMD_THRESHOLD && T::has_simd_support() {
        unsafe { add_simd(m1, m2) }
    } else {
        add_scalar(m1, m2)
    }
}

fn add_scalar<T>(m1: &Matrix<T>, m2: &Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    let data: Vec<T> = m1
        .data
        .iter()
        .zip(&m2.data)
        .map(|(a, b)| a.clone() + b.clone())
        .collect();

    Matrix::from_vec(m1.rows, m1.cols, data)
}

#[cfg(target_arch = "x86_64")]
unsafe fn add_simd<T>(m1: &Matrix<T>, m2: &Matrix<T>) -> Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    let total_elements = m1.rows * m1.cols;
    let mut result = Vec::with_capacity(total_elements);
    result.set_len(total_elements);

    result
        .par_chunks_mut(T::LANE_SIZE * 128)
        .zip(m1.data.par_chunks(T::LANE_SIZE * 128))
        .zip(m2.data.par_chunks(T::LANE_SIZE * 128))
        .for_each(|((r, a), b)| {
            let chunks = r.len() / T::LANE_SIZE;

            for i in 0..chunks {
                let offset = i * T::LANE_SIZE;

                // Prefetch data for future iterations
                if i + T::PREFETCH_DISTANCE < chunks {
                    let prefetch_offset = (i + T::PREFETCH_DISTANCE) * T::LANE_SIZE;

                    // Prefetch from both input matrices
                    T::prefetch(a[prefetch_offset..].as_ptr());
                    T::prefetch(b[prefetch_offset..].as_ptr());

                    // Optionally prefetch the result location (helpful for store operations)
                    T::prefetch(r[prefetch_offset..].as_ptr());
                }

                // Regular SIMD loading, addition, and storing
                let m1_vec = T::load(a[offset..].as_ptr());
                let m2_vec = T::load(b[offset..].as_ptr());
                let sum = <T as SimdOps>::add(m1_vec, m2_vec);
                T::store(r[offset..].as_mut_ptr(), sum);
            }

            let remaining_start = chunks * T::LANE_SIZE;
            for i in remaining_start..r.len() {
                r[i] = a[i].clone() + b[i].clone();
            }
        });

    Matrix::from_vec(m1.rows, m1.cols, result)
}

fn add_assign_matrix_impl<T>(m1: &mut Matrix<T>, m2: &Matrix<T>)
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + AddAssign<T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    // let total_elements = m1.rows * m1.cols;
    // TODO: implement add assign simd
    add_assign_scalar(m1, m2);
}

fn add_assign_scalar<T>(m1: &mut Matrix<T>, m2: &Matrix<T>)
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign<T> + Default,
{
    m1.data
        .iter_mut()
        .zip(&m2.data)
        .for_each(|(a, b)| *a += b.clone());
}

impl<T> Add for Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        add_matrix_impl(&self, &other)
    }
}

impl<T> Add for &Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    type Output = Matrix<T>;

    fn add(self, other: &Matrix<T>) -> Self::Output {
        add_matrix_impl(self, other)
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + AddAssign<T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    fn add_assign(&mut self, other: Matrix<T>) {
        assert_eq!(
            self.rows, other.rows,
            "Matrices must have the same number of rows"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrices must have the same number of columns"
        );

        add_assign_matrix_impl(self, &other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::{Complex32 as c32, Complex64 as c64};

    #[test]
    fn test_add_matrices() {
        let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = m1 + m2;
        assert_eq!(m3.rows(), 3);
        assert_eq!(m3.cols(), 3);
        assert_eq!(m3[(0, 0)], 2);
        assert_eq!(m3[(1, 1)], 6);
        assert_eq!(m3[(2, 2)], 10);
    }

    #[test]
    fn test_add_matrix_refs() {
        let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), 3);
        assert_eq!(m3.cols(), 3);
        assert_eq!(m3[(0, 0)], 2);
        assert_eq!(m3[(1, 1)], 6);
        assert_eq!(m3[(2, 2)], 10);
    }

    #[test]
    fn test_add_complex_matrix_refs() {
        let m1: Matrix<c32> = Matrix::new([[c32::new(2.0, 0.0); 3]; 3]);
        let m2: Matrix<c32> = Matrix::new([[c32::new(2.0, 0.0); 3]; 3]);

        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), 3);
        assert_eq!(m3.cols(), 3);
        assert_eq!(m3[(0, 0)], c32::new(4.0, 0.0)); // 2 + 2 = 4
        assert_eq!(m3[(1, 1)], c32::new(4.0, 0.0)); // 2 + 2 = 4
        assert_eq!(m3[(2, 2)], c32::new(4.0, 0.0)); // 2 + 2 = 4
    }

    #[test]
    fn test_add_assign_matrices() {
        let mut m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        m1 += m2;
        assert_eq!(m1.rows(), 3);
        assert_eq!(m1.cols(), 3);
        assert_eq!(m1[(0, 0)], 2);
        assert_eq!(m1[(1, 1)], 6);
        assert_eq!(m1[(2, 2)], 10);
    }

    #[test]
    fn test_add_large_matrix_c32() {
        let size = 1024;
        let data1 = vec![c32::new(1.0, 0.0); size * size];
        let data2 = vec![c32::new(2.0, 0.0); size * size];
        let m1 = Matrix::from_vec(size, size, data1);
        let m2 = Matrix::from_vec(size, size, data2);
        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), size);
        assert_eq!(m3.cols(), size);
        assert_eq!(m3[(0, 0)], c32::new(3.0, 0.0));
        assert_eq!(m3[(size - 1, size - 1)], c32::new(3.0, 0.0));

        // Test some random positions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let row = rng.gen_range(0..size);
            let col = rng.gen_range(0..size);
            assert_eq!(m3[(row, col)], c32::new(3.0, 0.0));
        }
    }

    #[test]
    fn test_add_large_matrix_c64() {
        let size = 1024;
        let data1 = vec![c64::new(1.0, 0.0); size * size];
        let data2 = vec![c64::new(2.0, 0.0); size * size];
        let m1 = Matrix::from_vec(size, size, data1);
        let m2 = Matrix::from_vec(size, size, data2);
        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), size);
        assert_eq!(m3.cols(), size);
        assert_eq!(m3[(0, 0)], c64::new(3.0, 0.0));
        assert_eq!(m3[(size - 1, size - 1)], c64::new(3.0, 0.0));

        // Test some random positions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let row = rng.gen_range(0..size);
            let col = rng.gen_range(0..size);
            assert_eq!(m3[(row, col)], c64::new(3.0, 0.0));
        }
    }
}
