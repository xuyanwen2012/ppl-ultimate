#pragma once

void demo_cpu_only(int grid_size);
void demo_gpu_only(int grid_size);
void demo_cpu_gpu_independent(int n_threads, int grid_size);
void demo_simple_cpu_gpu_coarse(int n_threads, int grid_size);
