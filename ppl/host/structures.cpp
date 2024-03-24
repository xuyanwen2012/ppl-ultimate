#include "shared/structures.h"

// Let's allocate 'capacity' instead of 'n_brt_nodes' for now
// Because usually n_brt_nodes is 99.x% of capacity

radix_tree::radix_tree(const size_t capacity) : capacity(capacity) {
  u_prefix_n = new uint8_t[capacity];
  u_has_leaf_left = new bool[capacity];
  u_has_leaf_right = new bool[capacity];
  u_left_child = new int[capacity];
  u_parent = new int[capacity];
}

radix_tree::~radix_tree() {
  delete[] u_prefix_n;
  delete[] u_has_leaf_left;
  delete[] u_has_leaf_right;
  delete[] u_left_child;
  delete[] u_parent;
}

octree::octree(const size_t capacity) : capacity(capacity) {
  u_children = new int[capacity][8];
  u_corner = new glm::vec4[capacity];
  u_cell_size = new float[capacity];
  u_child_node_mask = new int[capacity];
  u_child_leaf_mask = new int[capacity];
}

octree::~octree() {
  delete[] u_children;
  delete[] u_corner;
  delete[] u_cell_size;
  delete[] u_child_node_mask;
  delete[] u_child_leaf_mask;
}

constexpr auto educated_guess = 0.55;

pipe::pipe(const int n,
           const float min_coord,
           const float range,
           const int seed)
    : brt(n),
      oct(n * educated_guess),
      n_points(n),
      min_coord(min_coord),
      range(range),
      seed(seed) {
  u_points = new glm::vec4[n];
  u_morton = new morton_t[n];
  u_morton_alt = new morton_t[n];
  u_edge_count = new int[n];
  u_edge_offset = new int[n];
  // For CPU, no need to allocate the temporary storage
}

pipe::~pipe() {
  delete[] u_points;
  delete[] u_morton;
  delete[] u_morton_alt;
  delete[] u_edge_count;
  delete[] u_edge_offset;
}

void pipe::clearSmem() {
  // no effect on CPU
}