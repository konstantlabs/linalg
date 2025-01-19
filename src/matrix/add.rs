use super::mat::Matrix;
use aligned_vec::AVec;
use crossbeam_utils::CachePadded;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{
    ops::{Add, AddAssign, Mul, Sub},
    sync::{atomic::AtomicBool, Once},
};

static CPU_FEATURES: CachePadded<AtomicBool> = CachePadded::new(AtomicBool::new(false));
static CPU_FEATURES_INIT: Once = Once::new();

const PARALLEL_THRESHOLD: usize = 1024 * 64; // 64KB of f32 data
const SIMD_THRESHOLD: usize = 32; // Minimum elements for SIMD to be worth it
const CACHE_LINE_SIZE: usize = 64; // Common cache line size

trait SimdOps: Sized {
    type Vector;
    const LANE_SIZE: usize;

    #[cfg(target_arch = "x86_64")]
    unsafe fn load(ptr: *const Self) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn store(ptr: *mut Self, vec: Self::Vector);

    #[cfg(target_arch = "x86_64")]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    fn has_simd_support() -> bool;
}

// x86_64 implementations (AVX2)
#[cfg(target_arch = "x86_64")]
impl SimdOps for f32 {
    type Vector = __m256;
    const LANE_SIZE: usize = 8;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_load_ps(ptr)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_store_ps(ptr, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_ps(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for i32 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 8;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_load_si256(ptr as *const __m256i)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_store_si256(ptr as *mut __m256i, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_epi32(a, b)
    }
}

fn add_matrix_impl<'a, T>(m1: &'a Matrix<T>, m2: &'a Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default + SimdOps,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    // add_scalar(m1, m2)
    unsafe { add_simd(m1, m2) }
}

fn add_scalar<T>(m1: &Matrix<T>, m2: &Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default,
{
    let data: Vec<T> = m1
        .data
        .iter()
        .zip(&m2.data)
        .map(|(a, b)| a.clone() + b.clone())
        .collect();

    Matrix::from_vec(m1.rows, m1.cols, data)
}

#[cfg(target_arch = "x86_64")]
unsafe fn add_simd<T>(m1: &Matrix<T>, m2: &Matrix<T>) -> Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default + SimdOps,
{
    let total_elements = m1.rows * m1.cols;
    let mut result: Vec<T> = Vec::with_capacity(total_elements);
    result.set_len(total_elements);

    let chunks_size = T::LANE_SIZE;

    // Process SIMD chunks
    for i in 0..total_elements / chunks_size {
        let offset = i * chunks_size;
        let m1_vec = T::load(m1.data.as_ptr().add(offset));
        let m2_vec = T::load(m2.data.as_ptr().add(offset));
        let sum = <T as SimdOps>::add(m1_vec, m2_vec);
        T::store(result.as_mut_ptr().add(offset), sum);
    }

    // Process remaining elements
    let remaining_start = total_elements / chunks_size * chunks_size;
    for i in remaining_start..total_elements {
        result[i] = m1.data[i].clone() + m2.data[i].clone();
    }

    Matrix::from_vec(m1.rows, m1.cols, result)
}

impl<T> Add for Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default + SimdOps,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        add_matrix_impl(&self, &other)
    }
}

impl<T> Add for &Matrix<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default + SimdOps,
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
