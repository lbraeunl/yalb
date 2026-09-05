#ifndef DOMAIN_DECOMPOSITION_H
#define DOMAIN_DECOMPOSITION_H

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>

class Domain {
public:
    const int local_nx;
    const int x_offset;
    const int left_neighbor;
    const int right_neighbor;

    Domain(int nx, int ny, int rank, int size, bool periodic)
        : local_nx(
              nx / size
              + (rank < nx % size ? 1 : 0)),
          x_offset(
              rank * (nx / size)
              + std::min(rank, nx % size)),
          left_neighbor(
              rank > 0
                  ? rank - 1
                  : (periodic ? size - 1 : MPI_PROC_NULL)),
          right_neighbor(
              rank + 1 < size
                  ? rank + 1
                  : (periodic ? 0 : MPI_PROC_NULL))
    {
    }
};

class HaloBuffers {
public:
    explicit HaloBuffers(int ny);

private:
    friend void halo_exchange(
        const Kokkos::View<double***>& distribution,
        const Domain& domain,
        HaloBuffers& buffers);

    Kokkos::View<double*> send_left_;
    Kokkos::View<double*> send_right_;
    Kokkos::View<double*> receive_left_;
    Kokkos::View<double*> receive_right_;

#ifndef YALB_ENABLE_CUDA_AWARE_MPI
    Kokkos::View<double*, Kokkos::HostSpace> send_left_host_;
    Kokkos::View<double*, Kokkos::HostSpace> send_right_host_;
    Kokkos::View<double*, Kokkos::HostSpace> receive_left_host_;
    Kokkos::View<double*, Kokkos::HostSpace> receive_right_host_;
#endif
};

bool cuda_aware_mpi_is_enabled();

void halo_exchange(
    const Kokkos::View<double***>& distribution,
    const Domain& domain,
    HaloBuffers& buffers);

#endif // DOMAIN_DECOMPOSITION_H
