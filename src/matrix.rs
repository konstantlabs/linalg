use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Range, Sub};

use crate::matrix_view::MatrixView;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self {
        let data: Vec<T> = data.into_iter().flatten().collect();
        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }

    pub fn view(&'_ self, row_range: Range<usize>, col_range: Range<usize>) -> MatrixView<'_, T> {
        let start = row_range.start * self.cols + col_range.start;
        let end = row_range.end * self.cols + col_range.end;
        MatrixView::new(
            row_range.end - row_range.start,
            col_range.end - col_range.start,
            &self.data[start..end],
        )
    }

    fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col)
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.get_mut(row, col)
    }
}

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
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
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
            .zip(other.data.iter())
            .for_each(|(a, b)| *a += b.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_matrix() {
        let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(m.cols, 3);
        assert_eq!(m.rows, 3);
    }

    #[test]
    fn test_access_index() {
        let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(m[(0, 0)], 1);
    }

    #[test]
    fn test_write_index() {
        let mut m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        m[(0, 0)] = 10;
        assert_eq!(m[(0, 0)], 10);
    }

    #[test]
    fn test_view_matrix() {
        let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m_view = m.view(0..2, 0..2);
        assert_eq!(m_view.rows, 2);
        assert_eq!(m_view.cols, 2);
        assert_eq!(m_view[(0, 0)], 1);
    }

    #[test]
    fn test_add_two_matrices() {
        let m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = m1 + m2;
        assert_eq!(m3.rows, 3);
        assert_eq!(m3.cols, 3);
        assert_eq!(m3[(0, 0)], 2);
        assert_eq!(m3[(1, 1)], 6);
        assert_eq!(m3[(2, 2)], 10);
    }

    #[test]
    fn test_add_assign_two_matrices() {
        let mut m1: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        m1 += m2;
        assert_eq!(m1.rows, 3);
        assert_eq!(m1.cols, 3);
        assert_eq!(m1[(0, 0)], 2);
        assert_eq!(m1[(1, 1)], 6);
        assert_eq!(m1[(2, 2)], 10);
    }
}
