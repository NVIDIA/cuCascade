#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

void check(cudaError_t status) { if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << '\n'; std::exit(1); } }
int main() {
  check(cudaSetDevice(0));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  constexpr int iterations = 10000000;
  for (int trial = 0; trial < 7; ++trial) {
    for (int step = 0; step < 2; ++step) {
      int const mode = (trial + step) % 2;
      cudaStreamCaptureStatus status{};
      for (int i = 0; i < 10000; ++i) { check(cudaStreamIsCapturing(stream, &status)); }
      auto begin = std::chrono::steady_clock::now();
      if (mode == 0) {
        for (int i = 0; i < iterations; ++i) { check(cudaStreamIsCapturing(stream, &status)); }
      } else {
        for (int i = 0; i < iterations; ++i) { check(cudaStreamGetCaptureInfo(stream, &status)); }
      }
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
      std::cout << "{\"trial\":" << trial << ",\"api\":\"" << (mode == 0 ? "is_capturing" : "capture_info") << "\",\"ns_per_query\":" << static_cast<double>(ns)/iterations << ",\"capture_status\":" << status << "}\n";
    }
  }
  check(cudaStreamDestroy(stream));
}
