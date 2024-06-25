#pragma once

#include "shared/structures.h"

namespace cpu {

void dispatch_ComputeMorton(int n_threads, struct pipe* p, const int pool_id);

void dispatch_RadixSort(int n_threads, struct pipe* p, const int pool_id);

void dispatch_RemoveDuplicates(int n_threads, struct pipe* p, const int pool_id);

void dispatch_BuildRadixTree(int n_threads, struct pipe* p, const int pool_id);

void dispatch_EdgeCount(int n_threads, struct pipe* p, const int pool_id);

void dispatch_EdgeOffset(int n_threads, struct pipe* p, const int pool_id);

void dispatch_BuildOctree(int n_threads, struct pipe* p, const int pool_id);

}  // namespace cpu