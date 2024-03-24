
#pragma once

#include <cassert>
#include <glm/glm.hpp>
#include <stdexcept>

#include "defines.h"

// I am using only pointers because this gives me a unified front end for both
// CPU/and GPU
struct radix_tree {
  const size_t capacity;

  // ------------------------
  // Essential Data
  // ------------------------
  int n_brt_nodes = UNINITIALIZED;

  uint8_t *u_prefix_n;
  bool *u_has_leaf_left;
  bool *u_has_leaf_right;
  int *u_left_child;
  int *u_parent;

  // ------------------------
  // Constructors
  // ------------------------

  radix_tree() = delete;

  explicit radix_tree(size_t n_to_allocate);

  radix_tree(const radix_tree &) = delete;
  radix_tree &operator=(const radix_tree &) = delete;
  radix_tree(radix_tree &&) = delete;
  radix_tree &operator=(radix_tree &&) = delete;

  ~radix_tree();

  // ------------------------
  // Getter/Setters
  // ------------------------

  void set_n_nodes(const size_t n_nodes) {
    assert(n_nodes < capacity);
    n_brt_nodes = static_cast<int>(n_nodes);
  }

  [[nodiscard]] int n_nodes() const {
    if (n_brt_nodes == UNINITIALIZED)
      throw std::runtime_error("BRT nodes unset!!!");
    return n_brt_nodes;
  }
};

struct octree {
#define UNINITIALIZED (-1)

  const size_t capacity;

  // ------------------------
  // Essential Data
  // ------------------------

  int n_oct_nodes = UNINITIALIZED;

  // [Outputs]
  int (*u_children)[8];
  glm::vec4 *u_corner;
  float *u_cell_size;
  int *u_child_node_mask;
  int *u_child_leaf_mask;

  // ------------------------
  // Constructors
  // ------------------------

  octree() = delete;

  explicit octree(size_t capacity);

  octree(const octree &) = delete;
  octree &operator=(const octree &) = delete;
  octree(octree &&) = delete;
  octree &operator=(octree &&) = delete;

  ~octree();

  // ------------------------
  // Getter/Setters
  // ------------------------
  void set_n_nodes(const size_t n_nodes) {
    assert(n_nodes < capacity);
    n_oct_nodes = static_cast<int>(n_nodes);
  }

  [[nodiscard]] int n_nodes() const {
    if (n_oct_nodes == UNINITIALIZED)
      throw std::runtime_error("BRT nodes unset!!!");
    return n_oct_nodes;
  }
};
