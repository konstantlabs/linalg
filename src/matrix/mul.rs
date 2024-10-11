use super::mat::Matrix;
use std::ops::{Add, Mul, Sub};

impl<T> Mul<T> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: self.data.into_iter().map(|x| x * rhs.clone()).collect(),
        }
    }
}

// impl<T> Mul<Matrix<T>> for T
// where
//     T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
// {
//     type Output = Matrix<T>;

//     fn mul(self, rhs: Matrix<T>) -> Self::Output {
//         rhs * self
//     }
// }
