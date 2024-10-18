#include <sched.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

void print_cpu_info() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) {
    std::cerr << "Failed to open /proc/cpuinfo" << std::endl;
    return;
  }

  std::string line;
  while (std::getline(cpuinfo, line)) {
    // Output each line from /proc/cpuinfo
    std::cout << line << std::endl;
  }

  cpuinfo.close();
}

// Function to pin thread to a specific core
void pin_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

// 1. Monte Carlo Integration for Estimating Pi
double monte_carlo_pi(int points) {
  int inside_circle = 0;
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < points; ++i) {
    double x = dist(gen);
    double y = dist(gen);
    if (x * x + y * y <= 1.0) {
      inside_circle++;
    }
  }
  return (4.0 * inside_circle) / points;  // Estimate of pi
}

// 2. Matrix Multiplication Benchmark
void matrix_multiplication_benchmark(int size,
                                     vector<vector<double>>& matrixA,
                                     vector<vector<double>>& matrixB,
                                     vector<vector<double>>& result) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < size; ++k) {
        result[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
  }
}

// 3. Irregular Graph Traversal
void graph_traversal_benchmark(int nodes,
                               unordered_map<int, vector<int>>& graph,
                               int start_node,
                               int& visit_count) {
  unordered_set<int> visited;
  vector<int> stack = {start_node};
  visit_count = 0;

  while (!stack.empty()) {
    int node = stack.back();
    stack.pop_back();
    if (visited.find(node) == visited.end()) {
      visited.insert(node);
      ++visit_count;
      for (int neighbor : graph[node]) {
        if (visited.find(neighbor) == visited.end()) {
          stack.push_back(neighbor);
        }
      }
    }
  }
}

// Benchmark function for each core
void benchmark_core(int core_id,
                    int nodes,
                    double& float_time,
                    double& matrix_time,
                    double& graph_time) {
  pin_thread_to_core(core_id);

  // 1. Monte Carlo Integration Benchmark (Increased Points)le
  auto start = high_resolution_clock::now();
  monte_carlo_pi(1e5);  // Run with 100 million points for heavy load
  auto end = high_resolution_clock::now();
  float_time = duration<double, milli>(end - start).count();

  // 2. Matrix Multiplication Benchmark
  int size = 200;
  vector<vector<double>> matrixA(size, vector<double>(size, 1.5));
  vector<vector<double>> matrixB(size, vector<double>(size, 2.5));
  vector<vector<double>> result(size, vector<double>(size, 0.0));
  start = high_resolution_clock::now();
  matrix_multiplication_benchmark(size, matrixA, matrixB, result);
  end = high_resolution_clock::now();
  matrix_time = duration<double, milli>(end - start).count();

  // 3. Irregular Graph Traversal Benchmark (Increased Node and Connections)
  unordered_map<int, vector<int>> graph;
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(0, nodes - 1);
  for (int i = 0; i < nodes; ++i) {
    int connections =
        dis(gen) % 20 + 10;  // Increased to 10-30 connections per node
    for (int j = 0; j < connections; ++j) {
      graph[i].push_back(dis(gen));
    }
  }

  int visit_count = 0;
  start = high_resolution_clock::now();
  graph_traversal_benchmark(nodes, graph, 0, visit_count);
  end = high_resolution_clock::now();
  graph_time = duration<double, milli>(end - start).count();
}

int main() {
  std::cout << "CPU Information:" << std::endl;
  print_cpu_info();

  int num_cores = thread::hardware_concurrency();
  int nodes = 20000;  // Increased node count for the graph traversal benchmark
  vector<thread> threads(num_cores);
  vector<double> float_times(num_cores, 0), matrix_times(num_cores, 0),
      graph_times(num_cores, 0);

  // Run benchmark on each core
  for (int i = 0; i < num_cores; ++i) {
    threads[i] = thread(benchmark_core,
                        i,
                        nodes,
                        ref(float_times[i]),
                        ref(matrix_times[i]),
                        ref(graph_times[i]));
  }

  // Join threads
  for (int i = 0; i < num_cores; ++i) {
    threads[i].join();
  }

  // Display results in a table
  cout << left << setw(10) << "Core ID" << setw(20) << "Float Time (ms)"
       << setw(20) << "Matrix Time (ms)" << setw(20) << "Graph Time (ms)"
       << endl;
  cout << string(70, '-') << endl;
  for (int i = 0; i < num_cores; ++i) {
    cout << left << setw(10) << i << setw(20) << float_times[i] << setw(20)
         << matrix_times[i] << setw(20) << graph_times[i] << endl;
  }

  return 0;
}
