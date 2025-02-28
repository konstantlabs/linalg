use super::{mat::Matrix, simd::SimdOps};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

const SIMD_THRESHOLD: usize = 256 * 256; // Minimum elements for SIMD to be worth it

fn multiply_matrices_impl<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
    assert_eq!(lhs.cols(), rhs.rows(), "Incompatible matrix dimensions");

    let rows = lhs.rows();
    let cols = rhs.cols();
    let inner_dim = lhs.cols();
    let total_elements = rows * cols;

    // Use Strassen for large square matrices that are powers of 2
    if rows == cols && cols == inner_dim && rows >= 128 && (rows & (rows - 1)) == 0 {
        multiply_strassen(lhs, rhs)
    }
    // Use SIMD for moderately large matrices
    else if total_elements >= SIMD_THRESHOLD && inner_dim >= 16 && T::has_simd_support() {
        unsafe { multiply_simd(lhs, rhs) }
    }
    // Fall back to basic multiplication for small matrices
    else {
        multiply_scalar(lhs, rhs)
    }
}

fn multiply_strassen<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
    // Base case for small matrices or odd dimensions
    // Choose a threshold based on benchmarking - 64 is a good starting point
    if lhs.rows() <= 64 || lhs.rows() % 2 != 0 || lhs.cols() % 2 != 0 || rhs.cols() % 2 != 0 {
        if lhs.rows() * lhs.cols() >= SIMD_THRESHOLD && lhs.cols() >= 16 && T::has_simd_support() {
            // Use SIMD multiplication for large enough matrices
            unsafe {
                return multiply_simd(lhs, rhs);
            }
        } else {
            // Use scalar multiplication for small matrices
            return multiply_scalar(lhs, rhs);
        }
    }

    let n = lhs.rows() / 2;

    // Extract quadrants from input matrices
    let a11 = lhs.submatrix(0, 0, n, n);
    let a12 = lhs.submatrix(0, n, n, n);
    let a21 = lhs.submatrix(n, 0, n, n);
    let a22 = lhs.submatrix(n, n, n, n);

    let b11 = rhs.submatrix(0, 0, n, n);
    let b12 = rhs.submatrix(0, n, n, n);
    let b21 = rhs.submatrix(n, 0, n, n);
    let b22 = rhs.submatrix(n, n, n, n);

    // Compute the 7 Strassen products, potentially in parallel
    let p1_p2_p3_p4 = rayon::join(
        || {
            rayon::join(
                || multiply_strassen(&a11, &(&b12 - &b22)), // P1
                || multiply_strassen(&(&a11 + &a12), &b22), // P2
            )
        },
        || {
            rayon::join(
                || multiply_strassen(&(&a21 + &a22), &b11), // P3
                || multiply_strassen(&a22, &(&b21 - &b11)), // P4
            )
        },
    );

    let p5_p6_p7 = rayon::join(
        || {
            rayon::join(
                || multiply_strassen(&(&a11 + &a22), &(&b11 + &b22)), // P5
                || multiply_strassen(&(&a12 - &a22), &(&b21 + &b22)), // P6
            )
        },
        || multiply_strassen(&(&a11 - &a21), &(&b11 + &b12)), // P7
    );

    let ((p1, p2), (p3, p4)) = p1_p2_p3_p4;
    let ((p5, p6), p7) = p5_p6_p7;

    // Compute the output quadrants using our SIMD-accelerated addition/subtraction
    let c11 = &(&(&p5 + &p4) - &p2) + &p6;
    let c12 = &p1 + &p2;
    let c21 = &p3 + &p4;
    let c22 = &(&(&p5 + &p1) - &p3) - &p7;

    // Combine the quadrants into the final result
    let mut result = Matrix::empty(lhs.rows(), rhs.cols());

    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = c11[(i, j)].clone();
            result[(i, j + n)] = c12[(i, j)].clone();
            result[(i + n, j)] = c21[(i, j)].clone();
            result[(i + n, j + n)] = c22[(i, j)].clone();
        }
    }

    result
}

fn multiply_scalar<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    let rows = lhs.rows();
    let cols = rhs.cols();
    let inner_dim = lhs.cols();
    let mut result: Matrix<T> = Matrix::empty(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner_dim {
                result[(i, j)] = result[(i, j)].clone() + lhs[(i, k)].clone() * rhs[(k, j)].clone();
            }
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
unsafe fn multiply_simd<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
    let rows = lhs.rows();
    let cols = rhs.cols();
    let inner_dim = lhs.cols();
    let mut result = vec![T::default(); rows * cols];

    // Transpose rhs for better cache locality
    let rhs_transposed = rhs.transpose();

    result
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, row_chunk)| {
            for j in 0..cols {
                let mut sum = T::default();
                let lhs_row_start = i * inner_dim;
                let rhs_col_start = j * inner_dim; // Using transposed matrix

                // Process SIMD_LANE_SIZE elements at a time
                let simd_iterations = inner_dim / T::LANE_SIZE;
                for k in 0..simd_iterations {
                    let offset = k * T::LANE_SIZE;

                    // Prefetch data for future iterations
                    if k + T::PREFETCH_DISTANCE < simd_iterations {
                        let prefetch_offset = (k + T::PREFETCH_DISTANCE) * T::LANE_SIZE;
                        T::prefetch(&lhs.data[lhs_row_start + prefetch_offset] as *const T);
                        T::prefetch(
                            &rhs_transposed.data[rhs_col_start + prefetch_offset] as *const T,
                        );
                    }

                    let lhs_vec = T::load(&lhs.data[lhs_row_start + offset] as *const T);
                    let rhs_vec = T::load(&rhs_transposed.data[rhs_col_start + offset] as *const T);

                    // For matrix multiplication, we multiply corresponding elements and then sum them
                    let prod = <T as SimdOps>::mul(lhs_vec, rhs_vec);
                    sum = sum + T::horizontal_sum(prod);
                }

                // Handle remaining elements
                let remaining_start = simd_iterations * T::LANE_SIZE;
                for k in remaining_start..inner_dim {
                    sum = sum
                        + lhs.data[lhs_row_start + k].clone()
                            * rhs_transposed.data[rhs_col_start + k].clone();
                }

                row_chunk[j] = sum;
            }
        });

    Matrix::from_vec(rows, cols, result)
}

impl<T> Mul<T> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.into_iter().map(|x| x * rhs.clone()).collect(),
        }
    }
}

impl<T> Mul<&T> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &T) -> Self::Output {
        self * rhs.clone()
    }
}

impl<T> Mul<Matrix<T>> for Matrix<T>
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

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        multiply_matrices_impl(&self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for Matrix<T>
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

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        multiply_matrices_impl(&self, rhs)
    }
}

impl<T> Mul<Matrix<T>> for &Matrix<T>
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

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        multiply_matrices_impl(self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
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

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        multiply_matrices_impl(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_scalar_multiplication() {
        let m = Matrix::new([[1, 2], [3, 4]]);
        let result = m * 2;
        assert_eq!(result.data, &[2, 4, 6, 8]);
    }

    #[test]
    fn test_matrix_scalar_multiplication_dimensions() {
        let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
        let result = m * 3;
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_matrix_scalar_multiplication_float() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let result = m * 0.5;
        assert_eq!(result.data, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new([[1, 2], [3, 4]]);
        let m2 = Matrix::new([[5, 6], [7, 8]]);
        let result = m1 * m2;
        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_matrix_multiplication_dimensions() {
        let m1 = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::new([[7, 8], [9, 10], [11, 12]]);
        let result = m1 * m2;
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
    }

    #[test]
    #[should_panic(expected = "Incompatible matrix dimensions")]
    fn test_matrix_multiplication_incompatible_dimensions() {
        let m1 = Matrix::new([[1, 2], [3, 4]]);
        let m2 = Matrix::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let _ = m1 * m2;
    }

    #[test]
    fn test_matrix_ref_multiplication() {
        let m1 = Matrix::new([[1, 2], [3, 4]]);
        let m2 = Matrix::new([[5, 6], [7, 8]]);
        let result = &m1 * &m2;
        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_matrix_owned_ref_multiplication() {
        let m1 = Matrix::new([[1, 2], [3, 4]]);
        let m2 = Matrix::new([[5, 6], [7, 8]]);
        let result = m1 * &m2;
        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_matrix_ref_owned_multiplication() {
        let m1 = Matrix::new([[1, 2], [3, 4]]);
        let m2 = Matrix::new([[5, 6], [7, 8]]);
        let result = &m1 * m2;
        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_strassen_algorithm() {
        let size = 128; // Power of 2 for Strassen to work optimally
        let mut m1 = Matrix::<i32>::empty(size, size);
        let mut m2 = Matrix::<i32>::empty(size, size);

        // Fill matrices with some pattern
        for i in 0..size {
            for j in 0..size {
                m1[(i, j)] = (i * j % 10) as i32;
                m2[(i, j)] = ((i + j) % 10) as i32;
            }
        }

        // Compute using both methods to verify correctness
        let result_scalar = multiply_scalar(&m1, &m2);
        let result_strassen = multiply_strassen(&m1, &m2);

        assert_eq!(result_scalar, result_strassen);
    }

    #[test]
    fn test_strassen_vs_simd() {
        let size = 256; // Large enough to use both Strassen and SIMD
        let mut m1 = Matrix::<f32>::empty(size, size);
        let mut m2 = Matrix::<f32>::empty(size, size);

        // Fill matrices with some pattern
        for i in 0..size {
            for j in 0..size {
                m1[(i, j)] = (i * j % 10) as f32;
                m2[(i, j)] = ((i + j) % 10) as f32;
            }
        }

        // Compare result of Strassen vs SIMD
        let result_simd = unsafe { multiply_simd(&m1, &m2) };
        let result_strassen = multiply_strassen(&m1, &m2);

        // Check that results are approximately equal (floating point comparisons)
        for i in 0..size {
            for j in 0..size {
                assert!((result_simd[(i, j)] - result_strassen[(i, j)]).abs() < 1e-4);
            }
        }
    }
}
