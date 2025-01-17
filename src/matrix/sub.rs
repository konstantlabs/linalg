use super::mat::Matrix;
use std::ops::{Add, Mul, Sub, SubAssign};

fn sub_matrix_impl<'a, T>(m1: &'a Matrix<T>, m2: &'a Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    let data: Vec<T> = m1
        .data
        .iter()
        .zip(&m2.data)
        .map(|(a, b)| a.clone() - b.clone())
        .collect();

    Matrix::from_vec(m1.rows, m1.cols, data)
}

impl<T> Sub for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Self::Output {
        sub_matrix_impl(&self, &other)
    }
}

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn sub(self, other: &Matrix<T>) -> Self::Output {
        sub_matrix_impl(&self, &other)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_sub() {
        let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = m1 - m2;
        assert_eq!(m3.rows, 3);
        assert_eq!(m3.cols, 3);
        assert_eq!(m3[(0, 0)], 0);
        assert_eq!(m3[(1, 1)], 4);
        assert_eq!(m3[(2, 2)], 8);
    }

    #[test]
    fn test_matrix_sub_ref() {
        let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = &m1 - &m2;
        assert_eq!(m3.rows, 3);
        assert_eq!(m3.cols, 3);
        assert_eq!(m3[(0, 0)], 0);
        assert_eq!(m3[(1, 1)], 4);
        assert_eq!(m3[(2, 2)], 8);
    }

    #[test]
    fn test_matrix_subassign() {
        let mut m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        m1 -= m2;
        assert_eq!(m1.rows, 3);
        assert_eq!(m1.cols, 3);
        assert_eq!(m1[(0, 0)], 0);
        assert_eq!(m1[(1, 1)], 4);
        assert_eq!(m1[(2, 2)], 8);
    }
}
