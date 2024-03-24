#pragma once

#include "shared/structures.h"

namespace gpu {

void initialize_dispatcher(const int n_streams);
void release_dispatcher();
void sync_device();
void sync_stream(const int stream_id);

void dispatch_ComputeMorton(const int grid_size,
                            const int stream_id,
                            pipe& pipe);

void dispatch_RadixSort(const int grid_size, const int stream_id, pipe& pipe);

// void dispatch_RemoveDuplicates(const int grid_size,
//                                const int stream_id,
//                                pipe& pipe);

void dispatch_RemoveDuplicates_async(const int grid_size,
                                     const int stream_id,
                                     pipe& pipe);

void RemoveDuplicates_on_complete(const int grid_size,
                                  const int stream_id,
                                  pipe& pipe);

void dispatch_RemoveDuplicates_sync(const int grid_size,
                                    const int stream_id,
                                    pipe& pipe);

void dispatch_BuildRadixTree(const int grid_size,
                             const int stream_id,
                             pipe& pipe);

void dispatch_EdgeCount(const int grid_size, const int stream_id, pipe& pipe);

void dispatch_EdgeOffset_async(const int grid_size,
                               const int stream_id,
                               pipe& pipe);

void dispatch_BuildOctree(const int grid_size, const int stream_id, pipe& pipe);

}  // namespace gpu
