//! Runtime SIMD dispatch — commit 15.4.
//!
//! # Overview
//!
//! Detects CPU capabilities once at startup and selects the fastest available
//! kernel for each hot operation.  Callers write dispatch-agnostic code:
//!
//! ```rust
//! use llm_engine::simd::dispatch::global_kernels;
//!
//! let k = global_kernels();
//! println!("CPU: {}", k.features());
//! // let out = k.matmul(&a, &b)?;
//! // let norm = k.rmsnorm(&x, &weight, 1e-5)?;
//! // let probs = k.softmax(&logits)?;
//! ```
//!
//! # Kernel selection
//!
//! | Op       | x86_64 AVX2+FMA    | aarch64 (NEON)     | fallback     |
//! |----------|--------------------|---------------------|--------------|
//! | matmul   | `matmul_avx2`      | `matmul_blocked`    | `matmul_blocked` |
//! | rmsnorm  | `rmsnorm_simd`     | `rmsnorm_simd`      | `rmsnorm`    |
//! | softmax  | `softmax_simd`     | `softmax_simd`      | `softmax`    |
//!
//! `rmsnorm_simd` and `softmax_simd` are always selected on NEON because their
//! inner loops use `simd::dot` / `simd::scale_into`, which dispatch to NEON
//! automatically.  `matmul_avx2` falls back to `matmul_blocked` on aarch64
//! (the AVX2 `#[cfg]` block is compiled out), so `matmul_blocked` is the best
//! pure-Rust path there; the parallel NEON matmul is a Week-16 addition.

use std::fmt;
use std::sync::OnceLock;

use crate::ops::matmul::{matmul_avx2, matmul_blocked};
use crate::ops::norm::{rmsnorm, rmsnorm_simd};
use crate::ops::softmax::{softmax, softmax_dim};
use crate::ops::softmax_simd::{softmax_simd, softmax_simd_dim};
use crate::tensor::{Result, Tensor};

// ── CpuFeatures ───────────────────────────────────────────────────────────

/// CPU SIMD capabilities detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// x86_64: AVX2 256-bit integer / float SIMD.
    pub avx2: bool,
    /// x86_64: Fused multiply-add (256-bit).
    pub fma: bool,
    /// aarch64: NEON / AdvSIMD (always true on aarch64).
    pub neon: bool,
}

impl CpuFeatures {
    /// Probe the running CPU and return the detected feature set.
    ///
    /// This is cheap to call but involves `cpuid` on x86_64; prefer
    /// [`global_kernels`] which caches the result in a `OnceLock`.
    pub fn detect() -> Self {
        Self {
            avx2: Self::detect_avx2(),
            fma:  Self::detect_fma(),
            neon: Self::detect_neon(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx2() -> bool { is_x86_feature_detected!("avx2") }
    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx2() -> bool { false }

    #[cfg(target_arch = "x86_64")]
    fn detect_fma() -> bool { is_x86_feature_detected!("fma") }
    #[cfg(not(target_arch = "x86_64"))]
    fn detect_fma() -> bool { false }

    /// NEON is mandatory on all aarch64 targets per the AArch64 ABI.
    #[cfg(target_arch = "aarch64")]
    fn detect_neon() -> bool { true }
    #[cfg(not(target_arch = "aarch64"))]
    fn detect_neon() -> bool { false }

    /// True when the full AVX2 + FMA path is available.
    #[must_use]
    pub fn has_avx2_fma(&self) -> bool { self.avx2 && self.fma }

    /// True when NEON is available (always on aarch64).
    #[must_use]
    pub fn has_neon(&self) -> bool { self.neon }

    /// Human-readable summary line, e.g. `"aarch64 [NEON]"`.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.avx2 { parts.push("AVX2"); }
        if self.fma  { parts.push("FMA");  }
        if self.neon { parts.push("NEON"); }
        if parts.is_empty() { parts.push("scalar"); }

        #[cfg(target_arch = "x86_64")]   let arch = "x86_64";
        #[cfg(target_arch = "aarch64")]  let arch = "aarch64";
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let arch = "unknown";

        format!("{arch} [{}]", parts.join(", "))
    }
}

impl fmt::Display for CpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

// ── type aliases for the three kernel signatures ──────────────────────────

type MatmulFn   = fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>;
type RmsnormFn  = fn(&Tensor<f32>, &Tensor<f32>, f32) -> Result<Tensor<f32>>;
type SoftmaxFn  = fn(&Tensor<f32>) -> Result<Tensor<f32>>;
type SoftmaxDimFn = fn(&Tensor<f32>, usize) -> Result<Tensor<f32>>;

// ── Kernels ───────────────────────────────────────────────────────────────

/// Selected kernel set for the current CPU.
///
/// Obtain via [`global_kernels`] (cached) or [`Kernels::select`] (fresh).
pub struct Kernels {
    features:    CpuFeatures,
    matmul_fn:   MatmulFn,
    rmsnorm_fn:  RmsnormFn,
    softmax_fn:  SoftmaxFn,
    softmax_dim_fn: SoftmaxDimFn,
}

impl Kernels {
    /// Build a `Kernels` set from the given `CpuFeatures`.
    ///
    /// Call [`global_kernels`] instead — this is exposed for testing.
    #[must_use]
    pub fn select(f: CpuFeatures) -> Self {
        // matmul: AVX2+FMA kernel handles its own fallback internally via
        // the runtime `is_x86_feature_detected!` check, but we skip even
        // the function call overhead on non-x86 by pointing directly to
        // matmul_blocked on aarch64.
        let matmul_fn: MatmulFn = if f.has_avx2_fma() {
            matmul_avx2   // x86_64 with AVX2+FMA
        } else {
            matmul_blocked // aarch64 / scalar / older x86
        };

        // rmsnorm + softmax: SIMD variants use simd::dot / simd::scale_into
        // which already dispatch to NEON on aarch64, so we always prefer
        // the SIMD variant whenever any SIMD is present.
        let rmsnorm_fn: RmsnormFn = if f.has_avx2_fma() || f.has_neon() {
            rmsnorm_simd
        } else {
            rmsnorm
        };

        let softmax_fn: SoftmaxFn = if f.has_avx2_fma() || f.has_neon() {
            softmax_simd
        } else {
            softmax
        };

        let softmax_dim_fn: SoftmaxDimFn = if f.has_avx2_fma() || f.has_neon() {
            softmax_simd_dim
        } else {
            softmax_dim
        };

        Self { features: f, matmul_fn, rmsnorm_fn, softmax_fn, softmax_dim_fn }
    }

    /// The CPU features this kernel set was built for.
    #[must_use]
    pub fn features(&self) -> &CpuFeatures { &self.features }

    /// 2-D GEMM: `C = A @ B`.  Shape `[M,K] × [K,N] → [M,N]`.
    ///
    /// Dispatches to AVX2+FMA, NEON-backed blocked, or scalar blocked.
    #[inline]
    pub fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        (self.matmul_fn)(a, b)
    }

    /// RMSNorm over the last dimension of `x`.
    ///
    /// Dispatches to SIMD (NEON / AVX2) or scalar implementation.
    #[inline]
    pub fn rmsnorm(&self, x: &Tensor<f32>, weight: &Tensor<f32>, eps: f32) -> Result<Tensor<f32>> {
        (self.rmsnorm_fn)(x, weight, eps)
    }

    /// Softmax over the last dimension of `x`.
    #[inline]
    pub fn softmax(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        (self.softmax_fn)(x)
    }

    /// Softmax over a specific dimension `dim`.
    #[inline]
    pub fn softmax_dim(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>> {
        (self.softmax_dim_fn)(x, dim)
    }

    /// One-line description of the selected kernels, e.g.:
    /// `"aarch64 [NEON] | matmul=blocked rmsnorm=simd softmax=simd"`
    #[must_use]
    pub fn description(&self) -> String {
        let mm   = if self.features.has_avx2_fma() { "avx2"    } else { "blocked" };
        let norm = if self.features.has_avx2_fma() || self.features.has_neon() { "simd" } else { "scalar" };
        let smx  = if self.features.has_avx2_fma() || self.features.has_neon() { "simd" } else { "scalar" };
        format!("{} | matmul={mm} rmsnorm={norm} softmax={smx}", self.features)
    }
}

impl fmt::Debug for Kernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.description())
    }
}

// ── global singleton ──────────────────────────────────────────────────────

static GLOBAL_KERNELS: OnceLock<Kernels> = OnceLock::new();

/// Return a reference to the process-wide [`Kernels`] singleton.
///
/// CPU features are detected exactly once on first call (via [`OnceLock`]);
/// all subsequent calls are a single atomic load.
#[must_use]
pub fn global_kernels() -> &'static Kernels {
    GLOBAL_KERNELS.get_or_init(|| Kernels::select(CpuFeatures::detect()))
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ── CpuFeatures ───────────────────────────────────────────────────────

    #[test]
    fn test_detect_does_not_panic() {
        let _ = CpuFeatures::detect();
    }

    #[test]
    fn test_summary_is_non_empty() {
        let f = CpuFeatures::detect();
        assert!(!f.summary().is_empty());
    }

    #[test]
    fn test_display_matches_summary() {
        let f = CpuFeatures::detect();
        assert_eq!(format!("{f}"), f.summary());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_always_true_on_aarch64() {
        assert!(CpuFeatures::detect().neon);
        assert!(!CpuFeatures::detect().avx2);
        assert!(!CpuFeatures::detect().fma);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_summary_contains_neon_on_aarch64() {
        assert!(CpuFeatures::detect().summary().contains("NEON"));
    }

    // ── Kernels::select ───────────────────────────────────────────────────

    #[test]
    fn test_select_from_detected_features() {
        let f = CpuFeatures::detect();
        let k = Kernels::select(f);
        assert_eq!(*k.features(), f);
    }

    #[test]
    fn test_description_non_empty() {
        let k = Kernels::select(CpuFeatures::detect());
        assert!(!k.description().is_empty());
    }

    #[test]
    fn test_debug_format_is_description() {
        let k = Kernels::select(CpuFeatures::detect());
        assert_eq!(format!("{k:?}"), k.description());
    }

    // ── Kernels::matmul ───────────────────────────────────────────────────

    #[test]
    fn test_dispatch_matmul_2x2_identity() {
        let k  = Kernels::select(CpuFeatures::detect());
        let a  = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c  = k.matmul(&a, &id).unwrap();
        let expected = [1.0_f32, 2.0, 3.0, 4.0];
        for (g, e) in c.as_slice().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
    }

    #[test]
    fn test_dispatch_matmul_matches_naive() {
        use crate::ops::matmul::matmul_naive;
        let k = Kernels::select(CpuFeatures::detect());
        let (m, inner, n) = (32, 48, 24);
        let a = Tensor::from_vec(
            (0..(m * inner)).map(|i| i as f32 * 0.01).collect(), vec![m, inner],
        ).unwrap();
        let b = Tensor::from_vec(
            (0..(inner * n)).map(|i| (inner * n - i) as f32 * 0.005).collect(), vec![inner, n],
        ).unwrap();
        let expected = matmul_naive(&a, &b).unwrap();
        let got      = k.matmul(&a, &b).unwrap();
        assert_eq!(got.dims(), expected.dims());
        for (g, e) in got.as_slice().iter().zip(expected.as_slice()) {
            let rel = (g - e).abs() / e.abs().max(1.0);
            assert!(rel < 1e-3, "rel={rel:.2e}");
        }
    }

    // ── Kernels::rmsnorm ──────────────────────────────────────────────────

    #[test]
    fn test_dispatch_rmsnorm_no_nan() {
        let k = Kernels::select(CpuFeatures::detect());
        let x = Tensor::from_vec((0..16).map(|i| i as f32 * 0.1 + 0.1).collect(), vec![4, 4]).unwrap();
        let w = Tensor::ones(vec![4]);
        let out = k.rmsnorm(&x, &w, 1e-5).unwrap();
        assert!(out.as_slice().iter().all(|v| !v.is_nan() && !v.is_infinite()));
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_dispatch_rmsnorm_matches_scalar() {
        use crate::ops::norm::rmsnorm;
        let k = Kernels::select(CpuFeatures::detect());
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 + 0.1).collect();
        let x = Tensor::from_vec(data, vec![4, 8]).unwrap();
        let w = Tensor::ones(vec![8]);
        let scalar = rmsnorm(&x, &w, 1e-5).unwrap();
        let disp   = k.rmsnorm(&x, &w, 1e-5).unwrap();
        for (a, b) in disp.as_slice().iter().zip(scalar.as_slice()) {
            assert!((a - b).abs() < 1e-5, "dispatch={a}, scalar={b}");
        }
    }

    // ── Kernels::softmax ─────────────────────────────────────────────────

    #[test]
    fn test_dispatch_softmax_sums_to_one() {
        let k = Kernels::select(CpuFeatures::detect());
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let out = k.softmax(&x).unwrap();
        let s: f32 = out.as_slice().iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum={s}");
    }

    #[test]
    fn test_dispatch_softmax_dim_matches_scalar() {
        use crate::ops::softmax::softmax_dim;
        let k = Kernels::select(CpuFeatures::detect());
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let scalar = softmax_dim(&x, 0).unwrap();
        let disp   = k.softmax_dim(&x, 0).unwrap();
        for (a, b) in disp.as_slice().iter().zip(scalar.as_slice()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── global_kernels singleton ──────────────────────────────────────────

    #[test]
    fn test_global_kernels_returns_same_ref() {
        let a = global_kernels() as *const Kernels;
        let b = global_kernels() as *const Kernels;
        assert_eq!(a, b, "global_kernels must return the same address every call");
    }

    #[test]
    fn test_global_kernels_matmul_smoke() {
        let k = global_kernels();
        let a = Tensor::from_vec(vec![2.0_f32, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = k.matmul(&a, &b).unwrap();
        assert!((c.as_slice()[0] - 2.0).abs() < 1e-5);
        assert!((c.as_slice()[3] - 3.0).abs() < 1e-5);
    }
}
