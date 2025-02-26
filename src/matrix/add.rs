use super::mat::Matrix;
use num::complex::{Complex32 as c32, Complex64 as c64};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::ops::{Add, AddAssign, Mul, Sub};

const SIMD_THRESHOLD: usize = 512 * 512; // Minimum elements for SIMD to be worth it

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
        _mm256_loadu_ps(ptr)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_ps(ptr, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_ps(a, b)
    }
}

// x86_64 implementations (AVX2)
#[cfg(target_arch = "x86_64")]
impl SimdOps for f64 {
    type Vector = __m256d;
    const LANE_SIZE: usize = 8;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_loadu_pd(ptr)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_pd(ptr, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_pd(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for u32 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 8;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_si256(ptr as *mut __m256i, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_epi32(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for u64 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 4;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_si256(ptr as *mut __m256i, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_epi64(a, b)
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
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_si256(ptr as *mut __m256i, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_epi32(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for i64 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 4;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        _mm256_storeu_si256(ptr as *mut __m256i, vec)
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_add_epi64(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for c32 {
    type Vector = (__m256, __m256);
    const LANE_SIZE: usize = 4;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        let real_ptr = ptr as *const f32;
        let imag_ptr = real_ptr.add(1);

        let mut real_array = [0.0f32; 8];
        let mut imag_array = [0.0f32; 8];

        for i in 0..4 {
            real_array[i] = *real_ptr.add(i * 2);
            imag_array[i] = *imag_ptr.add(i * 2);
        }

        (
            _mm256_loadu_ps(real_array.as_ptr()),
            _mm256_loadu_ps(imag_array.as_ptr()),
        )
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        let mut real_array = [0.0f32; 8];
        let mut imag_array = [0.0f32; 8];

        _mm256_storeu_ps(real_array.as_mut_ptr(), vec.0);
        _mm256_storeu_ps(imag_array.as_mut_ptr(), vec.1);

        let real_ptr = ptr as *mut f32;
        let imag_ptr = real_ptr.add(1);

        for i in 0..4 {
            *real_ptr.add(i * 2) = real_array[i];
            *imag_ptr.add(i * 2) = imag_array[i];
        }
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        (_mm256_add_ps(a.0, b.0), _mm256_add_ps(a.1, b.1))
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for c64 {
    type Vector = (__m256d, __m256d);
    const LANE_SIZE: usize = 2;

    fn has_simd_support() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn load(ptr: *const Self) -> Self::Vector {
        let real_ptr = ptr as *const f64;
        let imag_ptr = real_ptr.add(1);

        let mut real_array = [0.0f64; 4];
        let mut imag_array = [0.0f64; 4];

        for i in 0..2 {
            real_array[i] = *real_ptr.add(i * 2);
            imag_array[i] = *imag_ptr.add(i * 2);
        }

        (
            _mm256_loadu_pd(real_array.as_ptr()),
            _mm256_loadu_pd(imag_array.as_ptr()),
        )
    }

    unsafe fn store(ptr: *mut Self, vec: Self::Vector) {
        let mut real_array = [0.0f64; 4];
        let mut imag_array = [0.0f64; 4];

        _mm256_storeu_pd(real_array.as_mut_ptr(), vec.0);
        _mm256_storeu_pd(imag_array.as_mut_ptr(), vec.1);

        let real_ptr = ptr as *mut f64;
        let imag_ptr = real_ptr.add(1);

        for i in 0..2 {
            *real_ptr.add(i * 2) = real_array[i];
            *imag_ptr.add(i * 2) = imag_array[i];
        }
    }

    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        (_mm256_add_pd(a.0, b.0), _mm256_add_pd(a.1, b.1))
    }
}

fn add_matrix_impl<'a, T>(m1: &'a Matrix<T>, m2: &'a Matrix<T>) -> Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Sync
        + Send,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    let total_elements = m1.rows * m1.cols;

    if total_elements >= SIMD_THRESHOLD && T::has_simd_support() {
        unsafe { add_simd(m1, m2) }
    } else {
        add_scalar(m1, m2)
    }
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
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    let total_elements = m1.rows * m1.cols;
    let mut result = Vec::with_capacity(total_elements);
    result.set_len(total_elements);

    result
        .par_chunks_mut(T::LANE_SIZE * 128)
        .zip(m1.data.par_chunks(T::LANE_SIZE * 128))
        .zip(m2.data.par_chunks(T::LANE_SIZE * 128))
        .for_each(|((r, a), b)| {
            let chunks = r.len() / T::LANE_SIZE;

            for i in 0..chunks {
                let offset = i * T::LANE_SIZE;
                let m1_vec = T::load(a[offset..].as_ptr());
                let m2_vec = T::load(b[offset..].as_ptr());
                let sum = <T as SimdOps>::add(m1_vec, m2_vec);
                T::store(r[offset..].as_mut_ptr(), sum);
            }

            let remaining_start = chunks * T::LANE_SIZE;
            for i in remaining_start..r.len() {
                r[i] = a[i].clone() + b[i].clone();
            }
        });

    Matrix::from_vec(m1.rows, m1.cols, result)
}

fn add_assign_matrix_impl<T>(m1: &mut Matrix<T>, m2: &Matrix<T>)
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + AddAssign
        + Default
        + SimdOps
        + Send
        + Sync,
{
    assert_eq!(
        m1.rows, m2.rows,
        "Matrices must have the same number of rows"
    );
    assert_eq!(
        m1.cols, m2.cols,
        "Matrices must have the same number of columns"
    );

    // let total_elements = m1.rows * m1.cols;
    // TODO: implement add assign simd
    add_assign_scalar(m1, m2);
}

fn add_assign_scalar<T>(m1: &mut Matrix<T>, m2: &Matrix<T>)
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign<T> + Default,
{
    m1.data
        .iter_mut()
        .zip(&m2.data)
        .for_each(|(a, b)| *a += b.clone());
}

impl<T> Add for Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        add_matrix_impl(&self, &other)
    }
}

impl<T> Add for &Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Default
        + SimdOps
        + Send
        + Sync,
{
    type Output = Matrix<T>;

    fn add(self, other: &Matrix<T>) -> Self::Output {
        add_matrix_impl(self, other)
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + AddAssign<T>
        + Default
        + SimdOps
        + Send
        + Sync,
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

        add_assign_matrix_impl(self, &other);
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
    fn test_add_two_complex_matrix_refs() {
        let m1: Matrix<c32> = Matrix::new([[c32::new(2.0, 0.0); 3]; 3]);
        let m2: Matrix<c32> = Matrix::new([[c32::new(2.0, 0.0); 3]; 3]);

        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), 3);
        assert_eq!(m3.cols(), 3);
        assert_eq!(m3[(0, 0)], c32::new(4.0, 0.0)); // 2 + 2 = 4
        assert_eq!(m3[(1, 1)], c32::new(4.0, 0.0)); // 2 + 2 = 4
        assert_eq!(m3[(2, 2)], c32::new(4.0, 0.0)); // 2 + 2 = 4
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

    #[test]
    fn test_large_matrix_complex32_add() {
        let size = 1024;
        let data1 = vec![c32::new(1.0, 0.0); size * size];
        let data2 = vec![c32::new(2.0, 0.0); size * size];
        let m1 = Matrix::from_vec(size, size, data1);
        let m2 = Matrix::from_vec(size, size, data2);
        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), size);
        assert_eq!(m3.cols(), size);
        assert_eq!(m3[(0, 0)], c32::new(3.0, 0.0));
        assert_eq!(m3[(size - 1, size - 1)], c32::new(3.0, 0.0));

        // Test some random positions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let row = rng.gen_range(0..size);
            let col = rng.gen_range(0..size);
            assert_eq!(m3[(row, col)], c32::new(3.0, 0.0));
        }
    }

    #[test]
    fn test_large_matrix_complex64_add() {
        let size = 1024;
        let data1 = vec![c64::new(1.0, 0.0); size * size];
        let data2 = vec![c64::new(2.0, 0.0); size * size];
        let m1 = Matrix::from_vec(size, size, data1);
        let m2 = Matrix::from_vec(size, size, data2);
        let m3 = &m1 + &m2;
        assert_eq!(m3.rows(), size);
        assert_eq!(m3.cols(), size);
        assert_eq!(m3[(0, 0)], c64::new(3.0, 0.0));
        assert_eq!(m3[(size - 1, size - 1)], c64::new(3.0, 0.0));

        // Test some random positions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let row = rng.gen_range(0..size);
            let col = rng.gen_range(0..size);
            assert_eq!(m3[(row, col)], c64::new(3.0, 0.0));
        }
    }
}
