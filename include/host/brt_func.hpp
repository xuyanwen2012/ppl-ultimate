#pragma once

#include "cuda_runtime.h"
#include "shared/morton_func.h"
#include "shared/structures.h"

namespace cpu {
#if defined(__GNUC__) || defined(__clang__)
#define CLZ(x) __builtin_clz(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define CLZ(x) _lzcnt_u32(x)
#else
#error "CLZ not supported on this platform"
#endif

inline unsigned int ceil_div_u32(const unsigned int a, const unsigned int b) {
  assert(b != 0);
  return (a + b - 1) / b;
}

inline uint8_t delta_u32(const unsigned int a, const unsigned int b) {
  [[maybe_unused]] constexpr unsigned int bit1_mask =
      static_cast<unsigned int>(1) << (sizeof(a) * 8 - 1);
  assert((a & bit1_mask) == 0);
  assert((b & bit1_mask) == 0);
  return static_cast<uint8_t>(CLZ(a ^ b) - 1);
}

inline int log2_ceil_u32(const unsigned int x) {
  // Counting from LSB to MSB, number of bits before last '1'
  // This is floor(log(x))
  const auto n_lower_bits = ((8 * sizeof(x)) - CLZ(x) - 1);

  // Add 1 if 2^n_lower_bits is less than x
  //     (i.e. we rounded down because x was not a power of 2)
  return static_cast<int>(n_lower_bits + ((1 << n_lower_bits) < x));
}

inline void process_radix_tree_i(const int i,
                                 const int n /*n_brt_nodes*/,
                                 const morton_t* codes,
                                 const radix_tree* out_brt) {
  // 'i' is the iterator within a chunk
  // 'codes' is the base address of the whole data, for each chunk, we need to
  // use the offset 'out_brt' is the base address of the whole data, for each
  // chunk, we need to use the offset

  const auto code_i = codes[i];

  const auto prefix_n = out_brt->u_prefix_n;
  const auto has_leaf_left = out_brt->u_has_leaf_left;
  const auto has_leaf_right = out_brt->u_has_leaf_right;
  const auto left_child = out_brt->u_left_child;
  const auto parent = out_brt->u_parents;

  // Determine direction of the range (+1 or -1)
  int d;
  if (i == 0) {
    d = 1;
  } else {
    const auto delta_diff_right = delta_u32(code_i, codes[i + 1]);
    const auto delta_diff_left = delta_u32(code_i, codes[i - 1]);
    const auto direction_difference = delta_diff_right - delta_diff_left;
    d = (direction_difference > 0) - (direction_difference < 0);
  }

  // Compute upper bound for the length of the range

  auto l = 0;
  if (i == 0) {
    // First node is root, covering whole tree
    l = n - 1;
  } else {
    const auto delta_min = delta_u32(code_i, codes[i - d]);
    auto l_max = 2;
    // Cast to ptrdiff_t so in case the result is negative (since d is +/- 1),
    // we can catch it and not index out of bounds
    while (i + static_cast<std::ptrdiff_t>(l_max) * d >= 0 &&
           i + l_max * d <= n &&
           delta_u32(code_i, codes[i + l_max * d]) > delta_min) {
      l_max *= 2;
    }
    const auto l_cutoff = (d == -1) ? i : n - i;
    int t;
    int divisor;
    // Find the other end using binary search
    for (t = l_max / 2, divisor = 2; t >= 1;
         divisor *= 2, t = l_max / divisor) {
      if (l + t <= l_cutoff &&
          delta_u32(code_i, codes[i + (l + t) * d]) > delta_min) {
        l += t;
      }
    }
  }

  const auto j = i + l * d;

  // Find the split position using binary search
  const auto delta_node = delta_u32(codes[i], codes[j]);
  prefix_n[i] = delta_node;
  auto s = 0;
  const auto max_divisor = 1 << log2_ceil_u32(l);
  auto divisor = 2;
  const auto s_cutoff = (d == -1) ? i - 1 : n - i - 1;
  for (auto t = ceil_div_u32(l, 2); divisor <= max_divisor;
       divisor <<= 1, t = ceil_div_u32(l, divisor)) {
    if (s + t <= s_cutoff &&
        delta_u32(code_i, codes[i + (s + t) * d]) > delta_node) {
      s += t;
    }
  }

  // Split position
  const auto gamma = i + s * d + std::min(d, 0);
  left_child[i] = gamma;
  has_leaf_left[i] = (std::min(i, j) == gamma);
  has_leaf_right[i] = (std::max(i, j) == gamma + 1);
  // Set parents of left and right children, if they aren't leaves
  // can't set this node as parent of its leaves, because the
  // leaf also represents an internal node with a differnent parent
  if (!has_leaf_left[i]) {
    parent[gamma] = i;
  }
  if (!has_leaf_right[i]) {
    parent[gamma + 1] = i;
  }
}
}  // namespace cpu
