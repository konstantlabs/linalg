use super::mat::Matrix;
use std::ops::{Add, AddAssign, Mul, Sub};

impl<T> Add for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Self::Output {
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
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix::from_vec(self.rows, self.cols, data)
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign<T>,
{
    fn add_assign(&mut self, other: Matrix<T>) {
        assert_eq!(
            self.rows(),
            other.rows(),
            "Matrices must have the same number of rows"
        );
        assert_eq!(
            self.cols(),
            other.cols(),
            "Matrices must have the same number of columns"
        );

        self.data
            .iter_mut()
            .zip(other.data)
            .for_each(|(a, b)| *a += b);
    }
}
