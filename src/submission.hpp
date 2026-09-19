#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <immintrin.h>

// ─── 1. Constants & Configuration ──────────────────────────────────────────

// Stencil weights: New = 0.5 * Center + 0.125 * (Sum of 4 neighbors)
constexpr double kCenterWeight   = 0.5;
constexpr double kNeighborWeight = 0.125;

// Pad each row with 3 doubles (24 bytes).
// Since row allocations are 64-byte aligned, col 0 is at offset 24 bytes,
// placing col 1 at offset 24 + 8 = 32 bytes (perfect 32-byte alignment).
// This allows interior stencil iterations to use fast aligned AVX2 loads (_mm256_load_pd).
constexpr std::size_t PREFIX_PADDING = 3;

// ─── 2. Memory Management ──────────────────────────────────────────────────

struct AlignedDeleter {
  void operator()(double* p) const noexcept {
    std::free(p);
  }
};
using AlignedBuffer = std::unique_ptr<double[], AlignedDeleter>;

inline AlignedBuffer make_aligned_buffer(std::size_t n) {
  auto* p = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
  std::memset(p, 0, n * sizeof(double));
  return AlignedBuffer(p);
}

// ─── 3. Non-owning Grid View (Span) ────────────────────────────────────────

template <typename T>
struct GridSpan {
  T*          ptr;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;
  bool        boundary_formed;

  // Returns pointer to row i, informing the compiler of its alignment characteristics
  T* row(std::size_t i) const noexcept {
    return static_cast<T*>(
        __builtin_assume_aligned(ptr + i * stride - PREFIX_PADDING, 64)) + PREFIX_PADDING;
  }
};

using ConstGridSpan   = GridSpan<const double>;
using MutableGridSpan = GridSpan<double>;

// ─── 4. Owning Grid Container ──────────────────────────────────────────────

class Grid {
private:
  std::size_t        rows_;
  std::size_t        cols_;
  std::size_t        stride_;
  AlignedBuffer      store_;
  mutable bool       boundary_formed_{false};

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows)
    , cols_(cols)
    // Add prefix padding to stride and round up to multiple of 8 doubles (64 bytes)
    , stride_((PREFIX_PADDING + cols + 7) & ~std::size_t(7))
    , store_(make_aligned_buffer(rows_ * stride_))
  {}

  // Raw data access (points to logical column 0)
  double*       data()       noexcept { return store_.get() + PREFIX_PADDING; }
  const double* data() const noexcept { return store_.get() + PREFIX_PADDING; }

  // Element access
  double& operator()(std::size_t i, std::size_t j)       { return data()[i * stride_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data()[i * stride_ + j]; }

  // Metadata
  std::size_t rows()   const noexcept { return rows_; }
  std::size_t cols()   const noexcept { return cols_; }
  std::size_t stride() const noexcept { return stride_; }

  // Boundary tracking (mutable to allow logical const operations)
  bool boundary_formed() const noexcept { return boundary_formed_; }
  void mark_boundary_formed() const noexcept { boundary_formed_ = true; }

  // View factories
  ConstGridSpan const_span() const noexcept {
    return {data(), rows_, cols_, stride_, boundary_formed_};
  }
  MutableGridSpan mutable_span() noexcept {
    return {data(), rows_, cols_, stride_, boundary_formed_};
  }
};

// ─── 5. Kernels ────────────────────────────────────────────────────────────

// Top and bottom boundaries are invariant across time steps.
// We only need to copy them once during the first step.
inline void copy_top_bottom(ConstGridSpan in, MutableGridSpan out) noexcept {
  if (in.boundary_formed) return;

  std::memcpy(out.row(0),           in.row(0),           in.cols * sizeof(double));
  std::memcpy(out.row(in.rows - 1), in.row(in.rows - 1), in.cols * sizeof(double));
}

// 5-point stencil evaluation for a single row using explicit AVX2 + FMA SIMD
inline __attribute__((target("avx2,fma"))) void stencil_row(
    const double* __restrict__ prev,
    const double* __restrict__ curr,
    const double* __restrict__ next,
    double*       __restrict__ dst,
    std::size_t cols) noexcept {
  std::size_t j = 1;

  const __m256d c0 = _mm256_set1_pd(kCenterWeight);
  const __m256d c1 = _mm256_set1_pd(kNeighborWeight);

  // Vectorized main loop: process 4 doubles per vector instruction
  for (; j + 3 < cols - 1; j += 4) {
    // Aligned loads (guaranteed by PREFIX_PADDING starting at col 1)
    __m256d p = _mm256_load_pd(&prev[j]);
    __m256d c = _mm256_load_pd(&curr[j]);
    __m256d n = _mm256_load_pd(&next[j]);

    // Unaligned loads for horizontal neighbors
    __m256d l = _mm256_loadu_pd(&curr[j - 1]);
    __m256d r = _mm256_loadu_pd(&curr[j + 1]);

    __m256d sum_neighbors = _mm256_add_pd(_mm256_add_pd(p, n), _mm256_add_pd(l, r));
    __m256d res = _mm256_fmadd_pd(c1, sum_neighbors, _mm256_mul_pd(c0, c));

    _mm256_store_pd(&dst[j], res);
  }

  // Scalar epilogue for leftover boundary columns
  for (; j < cols - 1; ++j) {
    dst[j] = kCenterWeight * curr[j]
           + kNeighborWeight * (prev[j] + next[j] + curr[j - 1] + curr[j + 1]);
  }
}

// ─── 6. Orchestration ──────────────────────────────────────────────────────

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  ConstGridSpan   in  = old_grid.const_span();
  MutableGridSpan out = new_grid.mutable_span();

  copy_top_bottom(in, out);

  const std::size_t rows = in.rows;
  const std::size_t cols = in.cols;

  // Row-parallel OpenMP distribution.
  // Left and right boundary copies are fused directly into each thread's row task.
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* prev = in.row(i - 1);
    const double* curr = in.row(i);
    const double* next = in.row(i + 1);
    double* dst = out.row(i);

    if (!in.boundary_formed) {
      dst[0]        = curr[0];
      dst[cols - 1] = curr[cols - 1];
    }

    stencil_row(prev, curr, next, dst, cols);
  }

  old_grid.mark_boundary_formed();
  new_grid.mark_boundary_formed();
}
