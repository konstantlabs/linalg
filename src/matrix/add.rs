use super::mat::Matrix;
use std::ops::{Add, AddAssign, Mul, Sub};

fn add_matrix_impl<'a, T>(m1: &'a Matrix<T>, m2: &'a Matrix<T>) -> Matrix<T>
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
        .map(|(a, b)| a.clone() + b.clone())
        .collect();

    Matrix::from_vec(m1.rows, m1.cols, data)
}

impl<T> Add for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        add_matrix_impl(&self, &other)
    }
}

impl<T> Add for &Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    type Output = Matrix<T>;

    fn add(self, other: &Matrix<T>) -> Self::Output {
        add_matrix_impl(self, other)
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign<T>,
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

        self.data
            .iter_mut()
            .zip(other.data)
            .for_each(|(a, b)| *a += b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_two_matrices() {
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
    fn test_add_two_matrix_refs() {
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
    fn test_add_assign_two_matrices() {
        let mut m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        m1 += m2;
        assert_eq!(m1.rows(), 3);
        assert_eq!(m1.cols(), 3);
        assert_eq!(m1[(0, 0)], 2);
        assert_eq!(m1[(1, 1)], 6);
        assert_eq!(m1[(2, 2)], 10);
    }
}
