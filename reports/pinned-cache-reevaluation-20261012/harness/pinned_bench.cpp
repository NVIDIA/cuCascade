#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>
#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using clock_type = std::chrono::steady_clock;
using allocator_type = cucascade::memory::small_pinned_host_memory_resource;
void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct resource_metrics {
 std::atomic<std::uint64_t> calls{0},large_calls{0},requested{0};
 std::array<std::atomic<std::uint64_t>,64> sizes{};
};
class counted_resource {
 public:
  allocator_type* mr;
  resource_metrics* stats;
  counted_resource(allocator_type& m, resource_metrics& metrics):mr(&m),stats(&metrics) {}
  void* allocate(::cuda::stream_ref s, std::size_t n, std::size_t a) {
    stats->calls.fetch_add(1,std::memory_order_relaxed); stats->requested.fetch_add(n,std::memory_order_relaxed);
    if(n>allocator_type::MAX_SLAB_SIZE) stats->large_calls.fetch_add(1,std::memory_order_relaxed);
    std::size_t k=0, v=n; while(v>1 && k+1<stats->sizes.size()){v>>=1; ++k;} stats->sizes[k].fetch_add(1,std::memory_order_relaxed);
    return mr->allocate(s,n,a);
  }
  void deallocate(::cuda::stream_ref s, void* p, std::size_t n, std::size_t a) noexcept {mr->deallocate(s,p,n,a);}
  void* allocate_sync(std::size_t n, std::size_t a) { auto* p=allocate(::cuda::stream_ref{cudaStream_t{nullptr}},n,a); check(cudaStreamSynchronize(nullptr)); return p; }
  void deallocate_sync(void* p, std::size_t n, std::size_t a) noexcept { mr->deallocate_sync(p,n,a); }
  bool operator==(counted_resource const& b) const noexcept{return this==&b;}
  friend void get_property(counted_resource const&, ::cuda::mr::host_accessible) noexcept{}
  friend void get_property(counted_resource const&, ::cuda::mr::device_accessible) noexcept{}
};
std::uint64_t digest(cudf::table_view table, rmm::cuda_stream_view s) {
 auto hashes=cudf::hashing::xxhash_64(table,0,s);
 std::vector<std::uint64_t> h(static_cast<std::size_t>(table.num_rows()));
 check(cudaMemcpyAsync(h.data(),hashes->view().data<std::uint64_t>(),h.size()*sizeof(std::uint64_t),cudaMemcpyDeviceToHost,s.value()));
 s.synchronize();
 std::uint64_t value=1469598103934665603ull;
 for(auto x:h){value^=x; value*=1099511628211ull;}
 return value;
}
int main(int argc,char** argv) try {
 if(argc<2) throw std::runtime_error("usage: pinned_bench parquet FILE [ITERATIONS] | churn [ITERATIONS]");
 check(cudaSetDevice(0));
 rmm::mr::cuda_async_memory_resource gpu_pool{256ull<<20,4ull<<30};
 auto prior_gpu=rmm::mr::set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>{std::move(gpu_pool)});
 rmm::mr::pinned_host_memory_resource pinned;
 cucascade::memory::fixed_size_host_memory_resource upstream{0,pinned,16ull<<20,16ull<<20,64ull<<10,16,1};
 allocator_type mr{upstream};
 resource_metrics metrics;
 counted_resource counted{mr,metrics};
 auto previous=cudf::set_pinned_memory_resource(counted);
 rmm::cuda_stream s{rmm::cuda_stream::flags::non_blocking};
 std::string mode=argv[1];
 std::cout<<"{\"type\":\"config\",\"candidate\":"
#ifdef PR_CANDIDATE
 <<"true"
#else
 <<"false"
#endif
 <<",\"pinned_threshold\":"<<cudf::get_allocate_host_as_pinned_threshold()<<",\"copy_threshold\":"<<cudf::get_kernel_pinned_copy_threshold()<<"}\n";
 if(mode=="parquet"){
  if(argc<3) throw std::runtime_error("missing parquet file");
  std::ifstream f(argv[2],std::ios::binary|std::ios::ate); if(!f) throw std::runtime_error("file open failed");
  auto n=f.tellg(); std::vector<char> bytes(static_cast<std::size_t>(n)); f.seekg(0); f.read(bytes.data(),n);
  int iterations=argc>3?std::stoi(argv[3]):11;
  auto options=cudf::io::parquet_reader_options::builder(cudf::io::source_info{cudf::host_span<char const>{bytes.data(),bytes.size()}}).build();
  std::uint64_t reference=0;
  for(int i=0;i<iterations;++i){
   auto c0=metrics.calls.load(std::memory_order_relaxed),l0=metrics.large_calls.load(std::memory_order_relaxed),b0=metrics.requested.load(std::memory_order_relaxed);
   auto start=clock_type::now();
   auto result=cudf::io::read_parquet(options,s.view()); s.synchronize();
   double ms=std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
   auto c1=metrics.calls.load(std::memory_order_relaxed),l1=metrics.large_calls.load(std::memory_order_relaxed),b1=metrics.requested.load(std::memory_order_relaxed);
   auto hash=digest(result.tbl->view(),s.view());
   if(i==0) reference=hash; else if(reference!=hash) throw std::runtime_error("parquet content digest changed");
   std::cout<<"{\"type\":\"parquet\",\"iteration\":"<<i<<",\"warm\":"<<(i>=3?"true":"false")<<",\"ms\":"<<ms<<",\"rows\":"<<result.tbl->num_rows()<<",\"columns\":"<<result.tbl->num_columns()<<",\"digest\":"<<hash<<",\"allocations\":"<<c1-c0<<",\"large_allocations\":"<<l1-l0<<",\"requested_bytes\":"<<b1-b0<<"}\n";
  }
 }else if(mode=="churn"){
  int iterations=argc>2?std::stoi(argv[2]):500;
  for(std::size_t bytes:{16384ull,131072ull,1048576ull,16777216ull}){
   for(int batch=0;batch<4;++batch){
    auto start=clock_type::now();
    for(int i=0;i<iterations;++i){
     void* ptr=mr.allocate(s.view(),bytes); s.synchronize();
     static_cast<volatile unsigned char*>(ptr)[0]=static_cast<unsigned char>(i);
     static_cast<volatile unsigned char*>(ptr)[bytes-1]=static_cast<unsigned char>(i);
     mr.deallocate(s.view(),ptr,bytes);
    }
    s.synchronize();
    double us=std::chrono::duration<double,std::micro>(clock_type::now()-start).count()/iterations;
    std::cout<<"{\"type\":\"churn\",\"bytes\":"<<bytes<<",\"batch\":"<<batch<<",\"warm\":"<<(batch>0?"true":"false")<<",\"us_per_op\":"<<us<<"}\n";
   }
   void* device=nullptr;check(cudaMalloc(&device,bytes));
   auto* ptr=static_cast<unsigned char*>(mr.allocate(s.view(),bytes));s.synchronize();std::memset(ptr,0xA5,bytes);
   check(cudaMemcpyAsync(device,ptr,bytes,cudaMemcpyHostToDevice,s.value()));mr.deallocate(s.view(),ptr,bytes);
   auto* reused=static_cast<unsigned char*>(mr.allocate(s.view(),bytes));s.synchronize();std::memset(reused,0x5A,bytes);
   std::vector<unsigned char> output(bytes);check(cudaMemcpyAsync(output.data(),device,bytes,cudaMemcpyDeviceToHost,s.value()));s.synchronize();
   if(!std::all_of(output.begin(),output.end(),[](auto x){return x==0xA5;})) throw std::runtime_error("H2D content corrupted");
   mr.deallocate(s.view(),reused,bytes);check(cudaFree(device));
  }
 }else throw std::runtime_error("unknown mode");
 check(cudaDeviceSynchronize());
 cudf::set_pinned_memory_resource(previous);
 std::cout<<"{\"type\":\"path\",\"size_log2_counts\":[";for(std::size_t i=0;i<metrics.sizes.size();++i){if(i)std::cout<<',';std::cout<<metrics.sizes[i].load(std::memory_order_relaxed);}std::cout<<"]}\n";
 rmm::mr::set_current_device_resource(std::move(prior_gpu));
 return 0;
}catch(std::exception const& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
