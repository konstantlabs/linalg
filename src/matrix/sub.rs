use super::mat::Matrix;
use std::ops::{Add, Mul, Sub, SubAssign};

impl<T> Sub for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Self::Output {
        assert_eq!(
            self.rows, other.rows,
            "Matrices must have the same number of rows"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrices must have the same number of columns"
        );

        let data: Vec<T> = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| a - b)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> SubAssign for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + SubAssign<T>,
{
    fn sub_assign(&mut self, other: Matrix<T>) {
        assert_eq!(
            self.rows, other.rows,
            "Matrices must have the same number of rows"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrices must have the same number of columns"
        );

        self.data
            .iter_mut()
            .zip(other.data)
            .for_each(|(a, b)| *a -= b);
    }
}
