#include <cuda.h>
#include <math.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <gcxx/api.hpp>
#include <mpicpp.hpp>
#include <nvtx3/nvtx3.hpp>

constexpr int number_of_warmups = 10;
constexpr int maxIt             = 100;


#define NCCL_CALL(stmt)                                               \
  do {                                                                \
    ncclResult_t result = (stmt);                                     \
    if (ncclSuccess != result) {                                      \
      fprintf(stderr, "[%s:%d] NCCL error: %s\n", __FILE__, __LINE__, \
              ncclGetErrorString(result));                            \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

inline int rowsinrank(int rank, int nranks, int N) {
  return N / nranks + ((N % nranks > rank) ? 1 : 0);
}

inline int startrow(int rank, int nranks, int N) {
  int remainder = N % nranks;
  int base_rows = N / nranks;

  // Each rank before `rank` gets base_rows, and an extra one if its index is
  // less than remainder
  return rank * base_rows + ((rank < remainder) ? rank : remainder);
}

#ifndef SKIP_CUDA_AWARENESS_CHECK
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
#error \
  "The used MPI Implementation does not have CUDA-aware support or CUDA-aware support can't be determined. Define SKIP_CUDA_AWARENESS_CHECK to skip this check."
#endif
#endif


using real = double;
#define NCCL_REAL_TYPE ncclDouble

__global__ void initialize_boundaries(real* __restrict__ const a_new,
                                      real* __restrict__ const a, const real pi,
                                      const int offset, const int N,
                                      const int my_ny);
void launch_initialize_boundaries(real* __restrict__ const a_new,
                                  real* __restrict__ const a, const real pi,
                                  const int offset, const int N,
                                  const int my_ny);

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new,
                              const real* __restrict__ const a,
                              const int iy_start, const int iy_end,
                              const int N);
void launch_jacobi_kernel(real* __restrict__ const a_new,
                          const real* __restrict__ const a, const int iy_start,
                          const int iy_end, const int N,
                          gcxx::StreamView stream);
void Halo_exchange(real* a, real* a_new, int N, const int top, int iy_end,
                   const int bottom, int iy_start, ncclComm_t,
                   gcxx::StreamView);

int main(int argc, char* argv[]) {
  auto mpi_env   = mpicxx::environment(argc, argv);
  auto worldcomm = mpicxx::comm::world();
  int rank       = worldcomm.rank();
  int nranks     = worldcomm.size();
  // CUDA_CALL(cudaGetDeviceCount(&nranks));
  int devcount = gcxx::Device::count();

  // CUDA_CALL(cudaSetDevice(rank % nranks));
  // CUDA_CALL(cudaFree(0));
  auto locdev = gcxx::Device::set(rank % devcount);

  ncclComm_t ncclcomm;
  ncclUniqueId id;
  if (rank == 0)
    NCCL_CALL(ncclGetUniqueId(&id));
  MPI_CALL(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_CALL(ncclCommInitRank(&ncclcomm, nranks, id, rank));
  worldcomm.ibarrier();

  int N = 1024;
  if (argc > 1) {
    N = atoi(argv[1]);
    if (N % 1024 != 0) {
      if (rank == 0)
        printf("size should be a multiple of 1024\n");
      mpi_env.~environment exit(EXIT_SUCCESS);
    }
  }
  {
    int chunk_size = rowsinrank(rank, nranks, N);
    auto a_raii =
      gcxx::memory::make_device_unique_ptr<real>(N * (chunk_size + 2));
    auto a_new_raii =
      gcxx::memory::make_device_unique_ptr<real>(N * (chunk_size + 2));
    gcxx::memory::Memset(a_raii, 0, N * (chunk_size + 2));
    gcxx::memory::Memset(a_new_raii, 0, N * (chunk_size + 2));
    real* a     = a_raii.get();
    real* a_new = a_new_raii.get();

    int iy_start_global = startrow(rank, nranks, N);
    int iy_start        = 1;
    int iy_end          = iy_start + chunk_size;

    gcxx::Stream inner_stream(gcxx::flags::streamType::SyncWithNull,
                              gcxx::flags::streamPriority::VeryLow);
    gcxx::Stream edge_stream(gcxx::flags::streamType::SyncWithNull,
                             gcxx::flags::streamPriority::Critical);

    gcxx::Event inner_done(gcxx::flags::eventCreate::disableTiming);
    gcxx::Event edge_done(gcxx::flags::eventCreate::disableTiming);

    const int top_pe = (rank + 1) % nranks;
    const int bot_pe = (rank + nranks - 1) % nranks;

    // Warmup NCCL+halo exchanges
    nvtxRangePushA("NCCL_Warmup");

    for (int i = 0; i < number_of_warmups; ++i) {
      Halo_exchange(a_new, a, N, top_pe, iy_end, bot_pe, iy_start, ncclcomm,
                    edge_stream);
      std::swap(a, a_new);
    }
    worldcomm.ibarrier();
    locdev.Synchronize();
    nvtxRangePop();

    // cudaGraph_t graphs[2];
    std::array<gcxx::Graph, 2> graphs;
    nvtxRangePushA("Graph_create");
    for (int g = 0; g < 2; ++g) {
      edge_stream.BeginCapture(gcxx::flags::streamCaptureMode::Global);

      // Launch edge-row Jacobi on edge_stream
      launch_jacobi_kernel(a_new, a, iy_start, iy_start + 1, N, edge_stream);
      launch_jacobi_kernel(a_new, a, iy_end - 1, iy_end, N, edge_stream);

      // NCCL halo exchange on edge_stream
      NCCL_CALL(ncclGroupStart());
      NCCL_CALL(
        ncclRecv(a_new, N, NCCL_REAL_TYPE, top_pe, ncclcomm, edge_stream));
      NCCL_CALL(ncclSend(a_new + (iy_end - 1) * N, N, NCCL_REAL_TYPE, bot_pe,
                         ncclcomm, edge_stream));
      NCCL_CALL(ncclRecv(a_new + iy_end * N, N, NCCL_REAL_TYPE, bot_pe,
                         ncclcomm, edge_stream));
      NCCL_CALL(ncclSend(a_new + iy_start * N, N, NCCL_REAL_TYPE, top_pe,
                         ncclcomm, edge_stream));
      NCCL_CALL(ncclGroupEnd());

      // Inner Jacobi on inner_stream
      launch_jacobi_kernel(a_new, a, iy_start + 1, iy_end - 1, N, inner_stream);

      graphs[g] = edge_stream.EndCapture();
      std::swap(a, a_new);
    }
    nvtxRangePop();

    // Instantiate graphs

    std::array<gcxx::GraphExec, 2> graph_exec;
    nvtxRangePushA("Graph_init");
    for (int g = 0; g < 2; ++g) {
      graph_exec[g] = graphs[g].Instantiate();
      graphs[g].destroy();
    }
    nvtxRangePop();

    // Warmup graph launches
    nvtxRangePushA("Graph_Warmup");
    for (int i = 0; i < 10; ++i) {
      graph_exec[0].Launch(inner_stream);
      graph_exec[1].Launch(inner_stream);
    }
    inner_stream.Synchronize();
    nvtxRangePop();

    // Initialize boundaries
    gcxx::memory::Memset(a_raii, 0, N * (chunk_size + 2));
    gcxx::memory::Memset(a_new_raii, 0, N * (chunk_size + 2));
    launch_initialize_boundaries(a, a_new, M_PI, iy_start_global - 1, N,
                                 chunk_size + 2);
    locdev.Synchronize();

    // Solve
    double start = MPI_Wtime();
    nvtxRangePushA("Jacobistep");
    for (int it = 0; it < maxIt; ++it) {
      graph_exec[it % 2].Launch(inner_stream);
    }
    nvtxRangePop();
    // CUDA_CALL(cudaDeviceSynchronize());
    locdev.Synchronize();
    double dur    = (MPI_Wtime() - start) / maxIt;
    double maxdur = 0;
    worldcomm.iallreduce(&dur, &maxdur, 1, mpicxx::op::max());

    if (rank == 0) {
      printf("NP %3d | LUPs %12d | perf %7.3f MLUPS/s\n", nranks, (N * N),
             N * N / maxdur / 1e6);
    }
  }
  // Cleanup
  NCCL_CALL(ncclCommDestroy(ncclcomm));
  return 0;
}

__global__ void initialize_boundaries(real* __restrict__ const a_new,
                                      real* __restrict__ const a, const real pi,
                                      const int offset, const int N,
                                      const int my_ny) {
  for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny;
       iy += blockDim.x * gridDim.x) {
    const real y0           = sin(2.0 * pi * (offset + iy) / (N - 1));
    a[iy * N + 0]           = y0;
    a[iy * N + (N - 1)]     = y0;
    a_new[iy * N + 0]       = y0;
    a_new[iy * N + (N - 1)] = y0;
  }
}

void launch_initialize_boundaries(real* __restrict__ const a_new,
                                  real* __restrict__ const a, const real pi,
                                  const int offset, const int N,
                                  const int my_ny) {
  // initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, N,
  //                                                 my_ny);
  // CUDA_CALL(cudaGetLastError());
  gcxx::launch::Kernel({128}, {my_ny / 128 + 1}, initialize_boundaries, a_new,
                       a, pi, offset, N, my_ny);
}

void launch_jacobi_kernel(real* __restrict__ const a_new,
                          const real* __restrict__ const a, const int iy_start,
                          const int iy_end, const int N,
                          gcxx::StreamView stream) {
  constexpr int dim_block_x = 32;
  constexpr int dim_block_y = 32;
  dim3 thread_dim(dim_block_x, dim_block_x);
  dim3 block_dim((N + dim_block_x - 1) / dim_block_x,
                 ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y);
  // jacobi_kernel<dim_block_x, dim_block_y>
  //   <<<block_dim, thread_dim, 0, stream>>>(a_new, a, iy_start, iy_end, N);
  // CUDA_CALL(cudaGetLastError());
  gcxx::launch::Kernel(stream, thread_dim, block_dim, 0,
                       jacobi_kernel<dim_block_x, dim_block_y>, a_new, a,
                       iy_start, iy_end, N);
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new,
                              const real* __restrict__ const a,
                              const int iy_start, const int iy_end,
                              const int N) {
  int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (iy < iy_end && ix < (N - 1)) {
    const real new_val = 0.25 * (a[iy * N + ix + 1] + a[iy * N + ix - 1] +
                                 a[(iy + 1) * N + ix] + a[(iy - 1) * N + ix]);
    a_new[iy * N + ix] = new_val;
  }
}

void Halo_exchange(real* a_new, real* a, int N, const int top, int iy_end,
                   const int bottom, int iy_start, ncclComm_t nccl_comm,
                   gcxx::StreamView edge_stream) {
  NCCL_CALL(ncclGroupStart());
  // clang-format off
  NCCL_CALL(ncclRecv(a_new                   , N, NCCL_REAL_TYPE, top   , nccl_comm, edge_stream.getRawStream()));
  NCCL_CALL(ncclSend(a_new + (iy_end - 1) * N, N, NCCL_REAL_TYPE, bottom, nccl_comm, edge_stream.getRawStream()));
  NCCL_CALL(ncclRecv(a_new + (iy_end * N)    , N, NCCL_REAL_TYPE, bottom, nccl_comm, edge_stream.getRawStream()));
  NCCL_CALL(ncclSend(a_new + (iy_start * N)  , N, NCCL_REAL_TYPE, top   , nccl_comm, edge_stream.getRawStream()));
  // clang-format on
  NCCL_CALL(ncclGroupEnd());
  // CUDA_CALL(cudaStreamSynchronize(edge_stream));
  edge_stream.Synchronize();
}
