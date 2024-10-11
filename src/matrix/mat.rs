use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Range, Sub};

use super::mat_view::MatrixView;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self {
        let data: Vec<T> = data.into_iter().flatten().collect();
        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }

    pub fn empty(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data length must match rows * cols"
        );
        Matrix { rows, cols, data }
    }

    pub fn view(&self, row_range: Range<usize>, col_range: Range<usize>) -> MatrixView<'_, T> {
        let start = row_range.start * self.cols + col_range.start;
        let end = row_range.end * self.cols + col_range.end;
        MatrixView::new(
            row_range.end - row_range.start,
            col_range.end - col_range.start,
            &self.data[start..end],
        )
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl<T> fmt::Display for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            write!(f, "[")?;
            for col in 0..self.cols {
                if col > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self[(row, col)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_matrix() {
        let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.rows(), 3);
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
    fn test_subtract_two_matrices() {
        let m1: Matrix<i32> = Matrix::new([[3, 2, 1], [6, 5, 4], [9, 8, 7]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m3 = m1 - m2;
        assert_eq!(m3.rows(), 3);
        assert_eq!(m3.cols(), 3);
        assert_eq!(m3[(0, 0)], 2);
        assert_eq!(m3[(1, 1)], 4);
        assert_eq!(m3[(2, 2)], 6);
    }

    #[test]
    fn test_subtract_assign_two_matrices() {
        let mut m1: Matrix<i32> = Matrix::new([[3, 2, 1], [6, 5, 4], [9, 8, 7]]);
        let m2: Matrix<i32> = Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        m1 -= m2;
        assert_eq!(m1.rows(), 3);
        assert_eq!(m1.cols(), 3);
        assert_eq!(m1[(0, 0)], 2);
        assert_eq!(m1[(1, 1)], 4);
        assert_eq!(m1[(2, 2)], 6);
    }

    #[test]
    fn test_display_matrix() {
        let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let expected = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n";
        assert_eq!(format!("{}", m), expected);
    }
}
