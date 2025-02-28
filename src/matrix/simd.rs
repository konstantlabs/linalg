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
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn broadcast(scalar: &Self) -> Self::Vector;

    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum(a: Self::Vector) -> Self;

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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_mul_ps(a, b)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_ps(*scalar)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        // Sum the 8 floats in the vector
        let sum1 = _mm256_hadd_ps(a, a); // [a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7]
        let sum2 = _mm256_hadd_ps(sum1, sum1); // [a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7]

        // Extract the lower 128 bits and upper 128 bits
        let low = _mm256_extractf128_ps(sum2, 0);
        let high = _mm256_extractf128_ps(sum2, 1);

        // Add the two 128-bit parts
        let result = _mm_add_ps(low, high);

        // Extract the lower 32 bits which contains our sum
        _mm_cvtss_f32(result)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_mul_pd(a, b)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_pd(*scalar)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        // Sum the 4 doubles in the vector
        let sum1 = _mm256_hadd_pd(a, a); // [a0+a1, a0+a1, a2+a3, a2+a3]

        // Extract the lower 128 bits and upper 128 bits
        let low = _mm256_extractf128_pd(sum1, 0); // [a0+a1, a0+a1]
        let high = _mm256_extractf128_pd(sum1, 1); // [a2+a3, a2+a3]

        // Add the two 128-bit parts
        let result = _mm_add_pd(low, high); // [a0+a1+a2+a3, a0+a1+a2+a3]

        // Extract the lower 64 bits which contains our sum
        _mm_cvtsd_f64(result)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_mullo_epi32(a, b)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_epi32(*scalar as i32)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        let mut result_array: [u32; 8] = [0; 8];
        _mm256_storeu_si256(result_array.as_mut_ptr() as *mut __m256i, a);

        result_array.iter().fold(0, |acc, &x| acc + x)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // AVX2 doesn't have a direct 64-bit integer multiplication
        // We need to extract, multiply, and repack
        let mut result_array: [u64; 4] = [0; 4];
        let mut a_array: [u64; 4] = [0; 4];
        let mut b_array: [u64; 4] = [0; 4];

        _mm256_storeu_si256(a_array.as_mut_ptr() as *mut __m256i, a);
        _mm256_storeu_si256(b_array.as_mut_ptr() as *mut __m256i, b);

        for i in 0..4 {
            result_array[i] = a_array[i] * b_array[i];
        }

        _mm256_loadu_si256(result_array.as_ptr() as *const __m256i)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_epi64x(*scalar as i64)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        let mut result_array: [u64; 4] = [0; 4];
        _mm256_storeu_si256(result_array.as_mut_ptr() as *mut __m256i, a);

        result_array.iter().fold(0, |acc, &x| acc + x)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm256_mullo_epi32(a, b)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_epi32(*scalar)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        let mut result_array: [i32; 8] = [0; 8];
        _mm256_storeu_si256(result_array.as_mut_ptr() as *mut __m256i, a);

        result_array.iter().fold(0, |acc, &x| acc + x)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // AVX2 doesn't have a direct 64-bit integer multiplication
        // We need to extract, multiply, and repack
        let mut result_array: [i64; 4] = [0; 4];
        let mut a_array: [i64; 4] = [0; 4];
        let mut b_array: [i64; 4] = [0; 4];

        _mm256_storeu_si256(a_array.as_mut_ptr() as *mut __m256i, a);
        _mm256_storeu_si256(b_array.as_mut_ptr() as *mut __m256i, b);

        for i in 0..4 {
            result_array[i] = a_array[i] * b_array[i];
        }

        _mm256_loadu_si256(result_array.as_ptr() as *const __m256i)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        _mm256_set1_epi64x(*scalar)
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        let mut result_array: [i64; 4] = [0; 4];
        _mm256_storeu_si256(result_array.as_mut_ptr() as *mut __m256i, a);

        result_array.iter().fold(0, |acc, &x| acc + x)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        let ac = _mm256_mul_ps(a.0, b.0); // ac
        let bd = _mm256_mul_ps(a.1, b.1); // bd
        let ad = _mm256_mul_ps(a.0, b.1); // ad
        let bc = _mm256_mul_ps(a.1, b.0); // bc

        let real = _mm256_sub_ps(ac, bd); // ac-bd (real part)
        let imag = _mm256_add_ps(ad, bc); // ad+bc (imaginary part)

        (real, imag)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        (_mm256_set1_ps(scalar.re), _mm256_set1_ps(scalar.im))
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        // Sum the real and imaginary parts separately
        let real_sum = {
            let sum1 = _mm256_hadd_ps(a.0, a.0);
            let sum2 = _mm256_hadd_ps(sum1, sum1);
            let low = _mm256_extractf128_ps(sum2, 0);
            let high = _mm256_extractf128_ps(sum2, 1);
            let result = _mm_add_ps(low, high);
            _mm_cvtss_f32(result)
        };

        let imag_sum = {
            let sum1 = _mm256_hadd_ps(a.1, a.1);
            let sum2 = _mm256_hadd_ps(sum1, sum1);
            let low = _mm256_extractf128_ps(sum2, 0);
            let high = _mm256_extractf128_ps(sum2, 1);
            let result = _mm_add_ps(low, high);
            _mm_cvtss_f32(result)
        };

        c32::new(real_sum, imag_sum)
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

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        let ac = _mm256_mul_pd(a.0, b.0); // ac
        let bd = _mm256_mul_pd(a.1, b.1); // bd
        let ad = _mm256_mul_pd(a.0, b.1); // ad
        let bc = _mm256_mul_pd(a.1, b.0); // bc

        let real = _mm256_sub_pd(ac, bd); // ac-bd (real part)
        let imag = _mm256_add_pd(ad, bc); // ad+bc (imaginary part)

        (real, imag)
    }

    unsafe fn broadcast(scalar: &Self) -> Self::Vector {
        (_mm256_set1_pd(scalar.re), _mm256_set1_pd(scalar.im))
    }

    unsafe fn horizontal_sum(a: Self::Vector) -> Self {
        // Sum the real and imaginary parts separately
        let real_sum = {
            let sum1 = _mm256_hadd_pd(a.0, a.0);
            let low = _mm256_extractf128_pd(sum1, 0);
            let high = _mm256_extractf128_pd(sum1, 1);
            let result = _mm_add_pd(low, high);
            _mm_cvtsd_f64(result)
        };

        let imag_sum = {
            let sum1 = _mm256_hadd_pd(a.1, a.1);
            let low = _mm256_extractf128_pd(sum1, 0);
            let high = _mm256_extractf128_pd(sum1, 1);
            let result = _mm_add_pd(low, high);
            _mm_cvtsd_f64(result)
        };

        c64::new(real_sum, imag_sum)
    }
}
