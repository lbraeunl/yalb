#include "domain_decomposition.h"

#include <stdexcept>

#if defined(YALB_ENABLE_CUDA_AWARE_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define YALB_HAS_MPI_CUDA_QUERY 1
#endif

namespace {

int halo_buffer_size(const int ny)
{
    if (ny <= 0) {
        throw std::invalid_argument("Halo buffer height must be positive");
    }
    return 3 * ny;
}

} // namespace


HaloBuffers::HaloBuffers(const int ny)
    : send_left_("send_left_halo", halo_buffer_size(ny)),
      send_right_("send_right_halo", halo_buffer_size(ny)),
      receive_left_("receive_left_halo", halo_buffer_size(ny)),
      receive_right_("receive_right_halo", halo_buffer_size(ny))
#ifndef YALB_ENABLE_CUDA_AWARE_MPI
      , send_left_host_("send_left_halo_host", halo_buffer_size(ny)),
      send_right_host_("send_right_halo_host", halo_buffer_size(ny)),
      receive_left_host_("receive_left_halo_host", halo_buffer_size(ny)),
      receive_right_host_("receive_right_halo_host", halo_buffer_size(ny))
#endif
{
}


bool cuda_aware_mpi_is_enabled()
{
#ifdef YALB_ENABLE_CUDA_AWARE_MPI
    return true;
#else
    return false;
#endif
}


bool cuda_aware_mpi_is_available()
{
#if defined(YALB_HAS_MPI_CUDA_QUERY) && defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() == 1;
#else
    return false;
#endif
}


void halo_exchange(
    const Kokkos::View<double***>& distribution,
    const Domain& domain,
    HaloBuffers& buffers)
{
    const int nx = distribution.extent_int(0);
    const int ny = distribution.extent_int(1);
    const int left_neighbor = domain.left_neighbor;
    const int right_neighbor = domain.right_neighbor;
    const int message_size = 3 * ny;

    if (buffers.send_left_.extent_int(0) != message_size) {
        throw std::invalid_argument(
            "Halo buffer height does not match distribution height");
    }

    // There is nothing to exchange for a single non-periodic subdomain.
    if (left_neighbor == MPI_PROC_NULL && right_neighbor == MPI_PROC_NULL) {
        return;
    }

    const auto send_left = buffers.send_left_;
    const auto send_right = buffers.send_right_;
    const auto receive_left = buffers.receive_left_;
    const auto receive_right = buffers.receive_right_;

    Kokkos::parallel_for(
        "pack_x_halos", Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            const int offset = 3 * j;
            send_left(offset) = distribution(1, j, 3);
            send_left(offset + 1) = distribution(1, j, 6);
            send_left(offset + 2) = distribution(1, j, 7);
            send_right(offset) = distribution(nx - 2, j, 1);
            send_right(offset + 1) = distribution(nx - 2, j, 5);
            send_right(offset + 2) = distribution(nx - 2, j, 8);
        });

    // MPI must not access a send buffer until the packing kernel has finished.
    Kokkos::fence();

#ifdef YALB_ENABLE_CUDA_AWARE_MPI
    // These pointers refer directly to the Kokkos device allocations. The
    // selected MPI implementation must support CUDA device pointers.
    MPI_Sendrecv(
        send_left.data(), message_size, MPI_DOUBLE, left_neighbor, 0,
        receive_right.data(), message_size, MPI_DOUBLE,
        right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        send_right.data(), message_size, MPI_DOUBLE, right_neighbor, 1,
        receive_left.data(), message_size, MPI_DOUBLE,
        left_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
    Kokkos::deep_copy(buffers.send_left_host_, send_left);
    Kokkos::deep_copy(buffers.send_right_host_, send_right);

    MPI_Sendrecv(
        buffers.send_left_host_.data(), message_size, MPI_DOUBLE,
        left_neighbor, 0,
        buffers.receive_right_host_.data(), message_size, MPI_DOUBLE,
        right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        buffers.send_right_host_.data(), message_size, MPI_DOUBLE,
        right_neighbor, 1,
        buffers.receive_left_host_.data(), message_size, MPI_DOUBLE,
        left_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (left_neighbor != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            receive_left, buffers.receive_left_host_);
    }
    if (right_neighbor != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            receive_right, buffers.receive_right_host_);
    }
#endif

    Kokkos::parallel_for(
        "unpack_x_halos", Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            const int offset = 3 * j;
            if (left_neighbor != MPI_PROC_NULL) {
                distribution(0, j, 1) = receive_left(offset);
                distribution(0, j, 5) = receive_left(offset + 1);
                distribution(0, j, 8) = receive_left(offset + 2);
            }
            if (right_neighbor != MPI_PROC_NULL) {
                distribution(nx - 1, j, 3) = receive_right(offset);
                distribution(nx - 1, j, 6) = receive_right(offset + 1);
                distribution(nx - 1, j, 7) = receive_right(offset + 2);
            }
        });
    Kokkos::fence();
}
