#include "shared/structures.h"

// Let's allocate 'capacity' instead of 'n_brt_nodes' for now
// Because usually n_brt_nodes is 99.x% of capacity

radix_tree::radix_tree(const size_t n_to_allocate) : capacity(n_to_allocate) {
  u_prefix_n = new uint8_t[n_to_allocate];
  u_has_leaf_left = new bool[n_to_allocate];
  u_has_leaf_right = new bool[n_to_allocate];
  u_left_child = new int[n_to_allocate];
  u_parent = new int[n_to_allocate];
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
