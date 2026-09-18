#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// ─── Grid ───────────────────────────────────────────────────────────────────
//
// Key changes from the starter:
//
//   1. **Aligned allocation** – the backing buffer is 64-byte aligned
//      (= one cache line = one AVX-512 register width).  This lets the
//      CPU load/store entire SIMD vectors without crossing cache-line
//      boundaries, which is measurably faster.
//
//   2. **Stride-padded rows** – each row is padded to a multiple of 8
//      doubles (64 bytes).  Combined with the aligned base address, this
//      guarantees *every* row starts at a 64-byte boundary.  The padding
//      columns (indices cols..stride-1) are never read or written by the
//      stencil — they just keep alignment perfect.
//
//   3. **Raw pointer + manual lifetime** – we use std::aligned_alloc
//      instead of std::vector so we control alignment.  The trade-off is
//      we must write a destructor and delete copy ops (the harness only
//      swaps pointers, so copies are never needed).
//
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;   // cols rounded up to next multiple of 8
  double*     store;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows)
    , cols_(cols)
    , stride_((cols + 7) & ~std::size_t(7))          // round up to mult-of-8
    , store(static_cast<double*>(  // convert void* to double* at compile time
          std::aligned_alloc(64, rows_ * stride_ * sizeof(double))))
  {
    // Zero the entire buffer (including padding) so nothing is uninitialized.
    std::memset(store, 0, rows_ * stride_ * sizeof(double));
  }

  ~Grid() { std::free(store); }

  // The harness never copies Grid objects — it swaps *pointers*.
  Grid(const Grid&)            = delete;
  Grid& operator=(const Grid&) = delete;

  // Move is cheap: just hand over the pointer.
  Grid(Grid&& o) noexcept
    : rows_(o.rows_), cols_(o.cols_), stride_(o.stride_), store(o.store)
  { o.store = nullptr; }

  // noexcept will let function never throw an exception
  // make operation more faster
  Grid& operator=(Grid&& o) noexcept {
    if (this != &o) {
      std::free(store);
      rows_ = o.rows_;  cols_ = o.cols_;  stride_ = o.stride_;
      store = o.store;   o.store = nullptr;
    }
    return *this;
  }

  // ── public interface (same signatures the harness expects) ──
  double& operator()(std::size_t i, std::size_t j)       { return store[i * stride_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return store[i * stride_ + j]; }
  std::size_t rows()   const { return rows_; }
  std::size_t cols()   const { return cols_; }
  std::size_t stride() const { return stride_; }
  const double* data() const { return store; }
  double*       data()       { return store; }
};

// ─── apply_stencil ─────────────────────────────────────────────────────────
//
// Three optimisations layered on top of each other:
//
//   1. OpenMP threading  – the tile grid is distributed across all
//      available cores with `#pragma omp parallel for`.
//
//   2. Cache blocking    – the interior is processed in BLOCK_R × BLOCK_C
//      tiles.  Each tile's read-set (3 rows × BLOCK_C doubles from
//      old_grid, plus 1 row × BLOCK_C doubles written to new_grid) fits
//      comfortably in each core's L2 cache (~1.25 MB), avoiding the
//      constant cache thrashing of a full-grid sweep.
//
//   3. SIMD hints        – `#pragma omp simd` on the inner loop tells
//      the compiler to emit AVX-512 vector instructions (8 doubles per
//      instruction).  The stride-aligned rows ensure every load/store
//      hits an aligned address.
//
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows   = old_grid.rows();
  const std::size_t cols   = old_grid.cols();
  const std::size_t stride = old_grid.stride();

  const double* __restrict__ in  = old_grid.data();
  double*       __restrict__ out = new_grid.data();

  // ── Boundary copy ──────────────────────────────────────────────────
  // Boundaries never change during diffusion; copy them unchanged.
  // memcpy is faster than element-wise loops for contiguous data.
  std::memcpy(out, in, cols * sizeof(double));                          // top row
  std::memcpy(out + (rows - 1) * stride,                               // bottom row
              in  + (rows - 1) * stride,
              cols * sizeof(double));

  for (std::size_t i = 1; i < rows - 1; ++i) {
    out[i * stride]            = in[i * stride];             // left col
    out[i * stride + cols - 1] = in[i * stride + cols - 1]; // right col
  }

  // ── Interior stencil (tiled + threaded + vectorised) ───────────────
  //
  //  Tile size rationale (BLOCK_R=64, BLOCK_C=256):
  //
  //    Read-set per tile  ≈ (64+2) × 256 × 8 B ≈ 132 KB
  //    Write-set per tile ≈  64    × 256 × 8 B ≈ 128 KB
  //    Total working set  ≈                       260 KB
  //
  //    L2 per core = 1.25 MB → tiles fit with room to spare.
  //
  constexpr std::size_t BLOCK_R = 64;
  constexpr std::size_t BLOCK_C = 256;

  #pragma omp parallel for schedule(static) collapse(2)
  for (std::size_t bi = 1; bi < rows - 1; bi += BLOCK_R) {
    for (std::size_t bj = 1; bj < cols - 1; bj += BLOCK_C) {
      // Clamp tile edges to the interior boundary.
      const std::size_t i_end = std::min(bi + BLOCK_R, rows - 1);
      const std::size_t j_end = std::min(bj + BLOCK_C, cols - 1);

      for (std::size_t i = bi; i < i_end; ++i) {
        const double* __restrict__ prev = in  + (i - 1) * stride;
        const double* __restrict__ curr = in  + i       * stride;
        const double* __restrict__ next = in  + (i + 1) * stride;
        double*       __restrict__ dst  = out + i       * stride;

        #pragma omp simd
        for (std::size_t j = bj; j < j_end; ++j) {
          dst[j] = 0.5   * curr[j]
                 + 0.125 * (prev[j] + next[j] + curr[j - 1] + curr[j + 1]);
        }
      }
    }
  }
}
