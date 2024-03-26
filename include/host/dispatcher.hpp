#pragma once

#include "shared/structures.h"

struct pipe;

// Note: CPU dispatcher is different from GPU dispatcher

namespace cpu {

// this will initialize a thread pool
void initialize_dispatcher();
void release_dispatcher();

void dispatch_ComputeMorton(int n_threads, struct pipe* p);

void dispatch_RadixSort(int n_threads, struct pipe* p);

void dispatch_RemoveDuplicates(int n_threads, struct pipe* p);

void dispatch_BuildRadixTree(int n_threads, struct pipe* p);

void dispatch_EdgeCount(int n_threads, struct pipe* p);

void dispatch_EdgeOffset(int n_threads, struct pipe* p);

void dispatch_BuildOctree(int n_threads, struct pipe* p);

}  // namespace cpu
