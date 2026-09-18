#pragma once

#include <cstddef>
#include <vector>


// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> store; // index: i * cols + j

public:
  Grid(std::size_t rows, std::size_t cols): rows_(rows), cols_(cols), store(rows * cols, 0.0){}
  double& operator()(std::size_t i, std::size_t j){
    return store[i * cols_ + j];
  }
  double  operator()(std::size_t i, std::size_t j) const{
    return store[i * cols_ + j ];
  }
  std::size_t rows() const{
    return rows_;
  }
  std::size_t cols() const{
    return cols_;
  }
  const double* data() const{
    return store.data();
  }
  double* data() {
    return store.data();
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  // copy boundary rows and cols
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  for(std::size_t j = 0; j < cols; j ++){
    new_grid(0, j) = old_grid(0, j); // top boundary
    new_grid(rows - 1, j) = old_grid(rows - 1, j); // bottom boundary
  }

  for (std::size_t i = 1; i < rows - 1; ++i) {
    new_grid(i, 0) = old_grid(i, 0);            // Left boundary 
    new_grid(i, cols - 1) = old_grid(i, cols - 1);     // Right boundary 
  }

  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* __restrict__ in_prev = old_grid.data() + (i - 1) * cols;
    const double* __restrict__ in_curr = old_grid.data() + i * cols;
    const double* __restrict__ in_next = old_grid.data() + (i + 1) * cols;
    double* __restrict__ out_curr = new_grid.data() + i * cols;

    for (std::size_t j = 1; j < cols - 1; ++j) {
      out_curr[j] = 0.5   * in_curr[j] + 
                    0.125 * (in_prev[j] + in_next[j] + in_curr[j - 1] + in_curr[j + 1]);
    }
  }
}
