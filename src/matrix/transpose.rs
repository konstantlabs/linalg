use super::mat::Matrix;
use std::ops::{Add, Mul, Sub};

impl<T> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    pub fn transpose(&self) -> Matrix<T> {
        let mut transposed_data = Vec::with_capacity(self.rows * self.cols);

        for col in 0..self.cols {
            for row in 0..self.rows {
                transposed_data.push(self.data[row * self.cols + col].clone());
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = Matrix::new([[1, 2], [3, 4], [5, 6]]);
        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(transposed.data, vec![1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn test_transpose_square_matrix() {
        let matrix = Matrix::new([[1, 2], [3, 4]]);

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.data, vec![1, 3, 2, 4]);
    }

    #[test]
    fn test_transpose_single_row_matrix() {
        let matrix = Matrix::new([[1, 2, 3]]);

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 1);
        assert_eq!(transposed.data, vec![1, 2, 3]);
    }
}
