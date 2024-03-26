#include <gtest/gtest.h>

// if on windows
#if defined(_WIN32) || defined(_WIN64)
#include <execution>
#define EXE_SEQ std::execution::seq,
#define EXE_PAR std::execution::par,
#else
#define EXE_SEQ
#define EXE_PAR
#endif

#include <glm/glm.hpp>
#include <iostream>
#include <memory>
#include <random>

#include "cuda/dispatcher.cuh"
#include "cuda/helper.cuh"
#include "host/brt_func.hpp"  // this is actually a CPU function
#include "shared/edge_func.h"
#include "shared/oct_func.h"
#include "shared/structures.h"

// ------------------ configs ------------------
// constexpr auto n = 1920 * 1080; // ~2M

constexpr auto n = 640 * 480;  // ~300k
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// used for quickly setting up the test
constexpr auto test_case_grid_size = 16;

[[nodiscard]] std::unique_ptr<struct pipe> generate_pipe() {
  auto gpu_pip = std::make_unique<struct pipe>(n, min_coord, range, seed);

  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(EXE_SEQ gpu_pip->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  return gpu_pip;
}

// ===============================================
//	Morton + Sort
// ===============================================

void test_morton_and_sort(const int grid_size) {
  auto gpu_pip = generate_pipe();
  const auto cpu_points = std::vector(gpu_pip->u_points, gpu_pip->u_points + n);

  // ------- testing region ------------
  constexpr auto stream_id = 0;
  gpu::dispatch_ComputeMorton(grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RadixSort(grid_size, stream_id, gpu_pip.get());
  gpu::sync_stream(stream_id);
  // -----------------------------------

  // generate CPU result
  std::vector<morton_t> cpu_morton(n);
  std::transform(EXE_PAR cpu_points.begin(),
                 cpu_points.end(),
                 cpu_morton.begin(),
                 [&](const auto& p) {
                   return shared::xyz_to_morton32(p, min_coord, range);
                 });
  std::sort(EXE_PAR cpu_morton.begin(), cpu_morton.end());

  const auto is_sorted =
      std::is_sorted(gpu_pip->u_morton, gpu_pip->u_morton + n);
  EXPECT_TRUE(is_sorted);

  const auto is_equal =
      std::equal(cpu_morton.begin(), cpu_morton.end(), gpu_pip->u_morton);
  EXPECT_TRUE(is_equal);
}

TEST(ComputeMorton, GridSize) {
  for (auto i = 1; i < 16; i++) {
    EXPECT_NO_FATAL_FAILURE(test_morton_and_sort(i));
  }
}

// ===============================================
//	Unique
// ===============================================

void test_unique(const int grid_size) {
  auto gpu_pip = generate_pipe();

  constexpr auto stream_id = 0;
  gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RadixSort(test_case_grid_size, stream_id, gpu_pip.get());
  SYNC_DEVICE();

  // generate CPU result (assume previous test is correct)
  const std::vector cpu_morton(gpu_pip->u_morton, gpu_pip->u_morton + n);

  // ------- testing region ------------
  gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, gpu_pip.get());
  gpu::sync_stream(stream_id);
  const auto gpu_n_unique = gpu_pip->n_unique_mortons();
  // -----------------------------------

  std::vector<morton_t> cpu_morton_alt(n);
  const auto last = std::unique_copy(
      cpu_morton.begin(), cpu_morton.end(), cpu_morton_alt.begin());
  const auto cpu_n_unique = std::distance(cpu_morton_alt.begin(), last);

  EXPECT_EQ(cpu_n_unique, gpu_n_unique);

  for (auto i = 0; i < gpu_n_unique; i++) {
    EXPECT_EQ(cpu_morton_alt[i], gpu_pip->u_morton_alt[i]);
  }
}

TEST(Unique, GridSize) {
  for (auto i = 1; i < 16; i++) {
    EXPECT_NO_FATAL_FAILURE(test_unique(i));
  }
}

// ===============================================
//	Binary Radix Tree
// ===============================================

void test_binary_radix_tree(const int grid_size) {
  auto gpu_pip = generate_pipe();

  constexpr auto stream_id = 0;
  gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RadixSort(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RemoveDuplicates_sync(
      test_case_grid_size, stream_id, gpu_pip.get());
  SYNC_DEVICE();

  const std::vector cpu_morton(gpu_pip->u_morton_alt,
                               gpu_pip->u_morton_alt + n);

  const auto cpu_n_brt_nodes = gpu_pip->n_brt_nodes();

  // ------- testing region ------------
  gpu::dispatch_BuildRadixTree(grid_size, stream_id, gpu_pip.get());
  gpu::sync_stream(stream_id);
  // -----------------------------------

  const radix_tree cpu_tree(cpu_n_brt_nodes);

  for (auto i = 0; i < cpu_n_brt_nodes; i++) {
    cpu::process_radix_tree_i(i, cpu_n_brt_nodes, cpu_morton.data(), &cpu_tree);
  }

  for (auto i = 0; i < cpu_n_brt_nodes; i++) {
    EXPECT_EQ(cpu_tree.u_prefix_n[i], gpu_pip->brt.u_prefix_n[i])
        << "Mismatch at index " << i;
    EXPECT_EQ(cpu_tree.u_has_leaf_left[i], gpu_pip->brt.u_has_leaf_left[i]);
    EXPECT_EQ(cpu_tree.u_has_leaf_right[i], gpu_pip->brt.u_has_leaf_right[i]);
    EXPECT_EQ(cpu_tree.u_left_child[i], gpu_pip->brt.u_left_child[i]);
    EXPECT_EQ(cpu_tree.u_parents[i], gpu_pip->brt.u_parents[i]);
  }
}

TEST(BinaryRadixTree, GridSize) {
  for (auto i = 1; i < 16; i++) {
    EXPECT_NO_FATAL_FAILURE(test_binary_radix_tree(i));
  }
}

// ===============================================
//	Edge Count + Offset
// ===============================================

void test_edge_count(const int grid_size) {
  auto gpu_pip = generate_pipe();

  constexpr auto stream_id = 0;
  gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RadixSort(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RemoveDuplicates_sync(
      test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_BuildRadixTree(test_case_grid_size, stream_id, gpu_pip.get());
  SYNC_DEVICE();

  // ------- testing region ------------
  gpu::dispatch_EdgeCount(grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_EdgeOffset(grid_size, stream_id, gpu_pip.get());
  gpu::sync_stream(stream_id);
  // -----------------------------------

  const auto cpu_n_brt_nodes = gpu_pip->n_brt_nodes();
  std::vector<int> cpu_edge_count(cpu_n_brt_nodes);
  std::vector<int> cpu_edge_offset(cpu_n_brt_nodes);

  for (auto i = 0; i < cpu_n_brt_nodes; i++) {
    shared::process_edge_count_i(i,
                                 gpu_pip->brt.u_prefix_n,
                                 gpu_pip->brt.u_parents,
                                 cpu_edge_count.data());
  }
  std::exclusive_scan(
      cpu_edge_count.begin(), cpu_edge_count.end(), cpu_edge_offset.begin(), 0);

  for (auto i = 0; i < cpu_n_brt_nodes; i++) {
    ASSERT_EQ(cpu_edge_count[i], gpu_pip->u_edge_counts[i])
        << "Mismatch at index " << i;
    ASSERT_EQ(cpu_edge_offset[i], gpu_pip->u_edge_offsets[i])
        << "Mismatch at index " << i;
  }
}

TEST(TestEdgeCount, GridSize) {
  for (auto i = 1; i < 16; i++) {
    EXPECT_NO_FATAL_FAILURE(test_edge_count(i));
  }
}

// ===============================================
//	Octree
// ===============================================

void cpu_build_octree(const struct pipe* p, const int n_brt_nodes) {
  const auto root_level = p->brt.u_prefix_n[0] / 3;
  const auto root_prefix =
      p->getSortedKeys()[0] >> (morton_bits - (3 * root_level));

  // compute root's corner
  shared::morton32_to_xyz(&p->oct.u_corner[0],
                          root_prefix << (morton_bits - (3 * root_level)),
                          min_coord,
                          range);
  p->oct.u_cell_size[0] = range;

  for (auto i = 1; i < n_brt_nodes; i++) {
    shared::process_oct_node(i,
                             p->oct.u_children,
                             p->oct.u_corner,
                             p->oct.u_cell_size,
                             p->oct.u_child_node_mask,
                             p->u_edge_offsets,
                             p->u_edge_counts,
                             p->getSortedKeys(),
                             p->brt.u_prefix_n,
                             p->brt.u_parents,
                             min_coord,
                             range);
  }
  // need to make this two steps
  for (auto i = 0; i < n_brt_nodes; i++) {
    shared::process_link_leaf(i,
                              p->oct.u_children,
                              p->oct.u_child_leaf_mask,
                              p->u_edge_offsets,
                              p->u_edge_counts,
                              p->getSortedKeys(),
                              p->brt.u_has_leaf_left,
                              p->brt.u_has_leaf_right,
                              p->brt.u_prefix_n,
                              p->brt.u_parents,
                              p->brt.u_left_child);
  }
}

void test_octree(const int grid_size) {
  auto gpu_pip = generate_pipe();

  constexpr auto stream_id = 0;
  gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RadixSort(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_RemoveDuplicates_sync(
      test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_BuildRadixTree(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_EdgeCount(test_case_grid_size, stream_id, gpu_pip.get());
  gpu::dispatch_EdgeOffset(test_case_grid_size, stream_id, gpu_pip.get());
  SYNC_DEVICE();

  // make a copy of the pipe (since the seed is the same, the result should be
  // the same)
  auto cpu_pip = std::make_unique<struct pipe>(n, min_coord, range, seed);
  std::copy_n(gpu_pip->u_points, n, cpu_pip->u_points);
  gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, cpu_pip.get());
  gpu::dispatch_RadixSort(test_case_grid_size, stream_id, cpu_pip.get());
  gpu::dispatch_RemoveDuplicates_sync(
      test_case_grid_size, stream_id, cpu_pip.get());
  gpu::dispatch_BuildRadixTree(test_case_grid_size, stream_id, cpu_pip.get());
  gpu::dispatch_EdgeCount(test_case_grid_size, stream_id, cpu_pip.get());
  gpu::dispatch_EdgeOffset(test_case_grid_size, stream_id, cpu_pip.get());
  SYNC_DEVICE();

  // ------- testing region ------------
  gpu::dispatch_BuildOctree(grid_size, stream_id, gpu_pip.get());
  gpu::sync_stream(stream_id);
  // -----------------------------------

  const auto n_brt_nodes = gpu_pip->n_brt_nodes();

  cpu_build_octree(cpu_pip.get(), n_brt_nodes);
}

TEST(TestOctree, GridSize) { EXPECT_NO_FATAL_FAILURE(test_octree(8)); }

int main(int argc, char** argv) {
  // some setups
  constexpr auto n_streams = 1;
  gpu::initialize_dispatcher(n_streams);

  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  gpu::release_dispatcher();
  return ret;
}
