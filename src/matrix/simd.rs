use num::complex::{Complex32 as c32, Complex64 as c64};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub trait SimdOps: Sized {
    type Vector;

    const LANE_SIZE: usize;
    const PREFETCH_DISTANCE: usize;

    #[cfg(target_arch = "x86_64")]
    unsafe fn load(ptr: *const Self) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn store(ptr: *mut Self, vec: Self::Vector);

    #[cfg(target_arch = "x86_64")]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn prefetch(ptr: *const Self) {
        // _MM_HINT_T0: Prefetch data into all levels of the cache hierarchy
        // Other options: _MM_HINT_T1, _MM_HINT_T2 (lower cache levels), _MM_HINT_NTA (non-temporal)
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    fn has_simd_support() -> bool;
}

// x86_64 implementations (AVX2)
#[cfg(target_arch = "x86_64")]
impl SimdOps for f32 {
    type Vector = __m256;
    const LANE_SIZE: usize = 8;
    const PREFETCH_DISTANCE: usize = 4;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_ps(a, b)
    }
}

// x86_64 implementations (AVX2)
#[cfg(target_arch = "x86_64")]
impl SimdOps for f64 {
    type Vector = __m256d;
    const LANE_SIZE: usize = 4;
    const PREFETCH_DISTANCE: usize = 6;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_pd(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for u32 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 8;
    const PREFETCH_DISTANCE: usize = 4;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_epi32(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for u64 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 4;
    const PREFETCH_DISTANCE: usize = 6;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_epi64(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for i32 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 8;
    const PREFETCH_DISTANCE: usize = 4;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_epi32(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for i64 {
    type Vector = __m256i;
    const LANE_SIZE: usize = 4;
    const PREFETCH_DISTANCE: usize = 6;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_sub_epi64(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for c32 {
    type Vector = (__m256, __m256);
    const LANE_SIZE: usize = 4;
    const PREFETCH_DISTANCE: usize = 6;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        (_mm256_sub_ps(a.0, b.0), _mm256_sub_ps(a.1, b.1))
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdOps for c64 {
    type Vector = (__m256d, __m256d);
    const LANE_SIZE: usize = 2;
    const PREFETCH_DISTANCE: usize = 8;

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

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        (_mm256_sub_pd(a.0, b.0), _mm256_sub_pd(a.1, b.1))
    }
}
