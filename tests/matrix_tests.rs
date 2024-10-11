use linalg::matrix::mat::Matrix;

#[test]
fn test_view_from_matrix() {
    let m: Matrix<i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let m_view = m.view(0..2, 0..2);
    assert_eq!(m_view.rows, 2);
    assert_eq!(m_view.cols, 2);
    assert_eq!(m_view[(0, 0)], 1);
}
