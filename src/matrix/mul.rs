use super::mat::Matrix;
use std::ops::{Add, Mul, Sub};

fn multiply_matrices<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    assert_eq!(lhs.cols(), rhs.rows(), "Incompatible matrix dimensions");

    let rows = lhs.rows();
    let cols = rhs.cols();
    let mut result: Matrix<T> = Matrix::empty(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..lhs.cols() {
                result[(i, j)] = result[(i, j)].clone() + lhs[(i, k)].clone() * rhs[(k, j)].clone();
            }
        }
    }

    result
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
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        multiply_matrices(&self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        multiply_matrices(&self, rhs)
    }
}

impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        multiply_matrices(self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        multiply_matrices(self, rhs)
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
}
