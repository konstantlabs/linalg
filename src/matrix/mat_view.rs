use std::ops::{Add, Index, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixView<'a, T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    pub rows: usize,
    pub cols: usize,
    data: &'a [T],
}

impl<'a, T> MatrixView<'a, T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    pub fn new(rows: usize, cols: usize, data: &'a [T]) -> Self {
        MatrixView { rows, cols, data }
    }

    fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixView<'a, T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_matrix_view() {
        let data = [1, 2, 3, 4];
        let view: MatrixView<i32> = MatrixView::new(2, 3, &data);
        assert_eq!(view.cols, 3);
        assert_eq!(view.rows, 2);
        assert_eq!(view.data, data);
    }

    #[test]
    fn test_get_element() {
        let data = [1, 2, 3, 4, 5, 6];
        let view = MatrixView::new(2, 3, &data);
        assert_eq!(*view.get(0, 0), 1);
        assert_eq!(*view.get(0, 2), 3);
        assert_eq!(*view.get(1, 1), 5);
    }

    #[test]
    fn test_index_operator() {
        let data = [1, 2, 3, 4, 5, 6];
        let view = MatrixView::new(2, 3, &data);
        assert_eq!(view[(0, 0)], 1);
        assert_eq!(view[(0, 2)], 3);
        assert_eq!(view[(1, 1)], 5);
    }

    #[test]
    fn test_matrix_equality() {
        let data = [1, 2, 3, 4];
        let view1 = MatrixView::new(2, 2, &data);
        let view2 = MatrixView::new(2, 2, &data);
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_inequality() {
        let data1 = [1, 2, 3, 4];
        let data2 = [1, 2, 3, 5];
        let view1 = MatrixView::new(2, 2, &data1);
        let view2 = MatrixView::new(2, 2, &data2);
        assert_ne!(view1, view2);
    }
}
