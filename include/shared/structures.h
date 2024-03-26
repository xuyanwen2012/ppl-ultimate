#pragma once

#include <cassert>
#include <glm/glm.hpp>
#include <stdexcept>

#include "defines.h"
#include "morton_func.h"

// I am using only pointers because this gives me a unified front end for both
// CPU/and GPU
struct radix_tree {
  const size_t capacity;

  // ------------------------
  // Essential Data
  // ------------------------
  int n_brt_nodes = UNINITIALIZED;

  uint8_t* u_prefix_n;
  bool* u_has_leaf_left;
  bool* u_has_leaf_right;
  int* u_left_child;
  int* u_parents;

  // ------------------------
  // Constructors
  // ------------------------

  radix_tree() = delete;

  explicit radix_tree(size_t n_to_allocate);

  radix_tree(const radix_tree&) = delete;
  radix_tree& operator=(const radix_tree&) = delete;
  radix_tree(radix_tree&&) = delete;
  radix_tree& operator=(radix_tree&&) = delete;

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
  const size_t capacity;

  // ------------------------
  // Essential Data
  // ------------------------

  int n_oct_nodes = UNINITIALIZED;

  // [Outputs]
  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;

  // ------------------------
  // Constructors
  // ------------------------

  octree() = delete;

  explicit octree(size_t capacity);

  octree(const octree&) = delete;
  octree& operator=(const octree&) = delete;
  octree(octree&&) = delete;
  octree& operator=(octree&&) = delete;

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
      throw std::runtime_error("OCT nodes unset!!!");
    return n_oct_nodes;
  }
};

struct pipe {
  // ------------------------
  // Essential Data (CPU/GPU shared)
  // ------------------------

  // mutable
  int n_unique = UNINITIALIZED;

  glm::vec4* u_points;
  morton_t* u_morton;
  morton_t* u_morton_alt;  // also used as the unique morton
  radix_tree brt;
  int* u_edge_counts;
  int* u_edge_offsets;
  octree oct;

  // read-only
  int n_points;
  float min_coord;
  float range;
  int seed;

  // ------------------------
  // Temporary Storage (for GPU only)
  // only allocated when GPU is used
  // ------------------------

  static constexpr auto RADIX = 256;
  static constexpr auto RADIX_PASSES = 4;
  static constexpr auto BIN_PART_SIZE = 7680;
  static constexpr auto GLOBAL_HIST_THREADS = 128;
  static constexpr auto BINNING_THREADS = 512;

  size_t binning_blocks;

  struct {
    unsigned int* d_global_histogram;
    unsigned int* d_index;
    unsigned int* d_first_pass_histogram;
    unsigned int* d_second_pass_histogram;
    unsigned int* d_third_pass_histogram;
    unsigned int* d_fourth_pass_histogram;
    int* u_flag_heads;
  } im_storage;

  // ------------------------
  // Constructors
  // ------------------------

  pipe() = delete;

  explicit pipe(int n_points,
                float min_coord = 0.0f,
                float range = 1024.0f,
                int seed = 114514);

  pipe(const pipe&) = delete;
  pipe& operator=(const pipe&) = delete;
  pipe(pipe&&) = delete;
  pipe& operator=(pipe&&) = delete;

  ~pipe();

  // ------------------------
  // Accessors (preffered over direct access)
  // ------------------------
  [[nodiscard]] int n_input() const { return n_points; }
  [[nodiscard]] int n_brt_nodes() const { return brt.n_nodes(); }

  [[nodiscard]] int n_unique_mortons() const {
    if (n_unique == UNINITIALIZED)
      throw std::runtime_error("Unique mortons unset!!!");
    return n_unique;
  }

  [[nodiscard]] int n_oct_nodes() const { return oct.n_nodes(); }

  void set_n_unique(const size_t n_unique) {
    assert(n_unique <= n_points);
    this->n_unique = static_cast<int>(n_unique);
  }

  // alias to make the code more understand able
  [[nodiscard]] const morton_t* getSortedKeys() const { return u_morton; }
  [[nodiscard]] morton_t* getUniqueKeys() { return u_morton_alt; }
  [[nodiscard]] const morton_t* getUniqueKeys() const { return u_morton_alt; }

  void clearSmem();
};
