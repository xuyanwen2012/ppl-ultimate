#include <cub/cub.cuh>

#include "cuda/helper.cuh"
#include "shared/structures.h"

// Let's allocate 'capacity' instead of 'n_brt_nodes' for now
// Because usually n_brt_nodes is 99.x% of capacity

radix_tree::radix_tree(const size_t capacity) : capacity(capacity) {
  MALLOC_MANAGED(&u_prefix_n, capacity);
  MALLOC_MANAGED(&u_has_leaf_left, capacity);
  MALLOC_MANAGED(&u_has_leaf_right, capacity);
  MALLOC_MANAGED(&u_left_child, capacity);
  MALLOC_MANAGED(&u_parent, capacity);
}

radix_tree::~radix_tree() {
  CUDA_FREE(u_prefix_n);
  CUDA_FREE(u_has_leaf_left);
  CUDA_FREE(u_has_leaf_right);
  CUDA_FREE(u_left_child);
  CUDA_FREE(u_parent);
}

octree::octree(const size_t capacity) : capacity(capacity) {
  MALLOC_MANAGED(&u_children, capacity * 8);
  MALLOC_MANAGED(&u_corner, capacity);
  MALLOC_MANAGED(&u_cell_size, capacity);
  MALLOC_MANAGED(&u_child_node_mask, capacity);
  MALLOC_MANAGED(&u_child_leaf_mask, capacity);
}

octree::~octree() {
  CUDA_FREE(u_children);
  CUDA_FREE(u_corner);
  CUDA_FREE(u_cell_size);
  CUDA_FREE(u_child_node_mask);
  CUDA_FREE(u_child_leaf_mask);
}

constexpr auto educated_guess = 0.55;

pipe::pipe(const int n)
    : n_points(n),
      brt(n),
      oct(n * educated_guess),
      binning_blocks(cub::DivideAndRoundUp(n, BIN_PART_SIZE)) {
  MALLOC_MANAGED(&u_points, n);
  MALLOC_MANAGED(&u_morton, n);
  MALLOC_MANAGED(&u_morton_alt, n);
  MALLOC_MANAGED(&u_edge_count, n);
  MALLOC_MANAGED(&u_edge_offset, n);

  MALLOC_DEVICE(&im_storage.d_global_histogram, RADIX * RADIX_PASSES);
  MALLOC_DEVICE(&im_storage.d_index, RADIX_PASSES);
  MALLOC_DEVICE(&im_storage.d_first_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_second_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_third_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);

  MALLOC_MANAGED(&im_storage.u_flag_heads, n);

  SYNC_DEVICE();
}

pipe::~pipe() {
  CUDA_FREE(u_points);
  CUDA_FREE(u_morton);
  CUDA_FREE(u_morton_alt);
  CUDA_FREE(u_edge_count);
  CUDA_FREE(u_edge_offset);

  CUDA_FREE(im_storage.d_global_histogram);
  CUDA_FREE(im_storage.d_index);
  CUDA_FREE(im_storage.d_first_pass_histogram);
  CUDA_FREE(im_storage.d_second_pass_histogram);
  CUDA_FREE(im_storage.d_third_pass_histogram);
  CUDA_FREE(im_storage.d_fourth_pass_histogram);

  CUDA_FREE(im_storage.u_flag_heads);
}