#include "domain_decomposition.h"

#include <stdexcept>


void halo_exchange(const Kokkos::View<double***>& distribution, const Domain& domain)
{
    const int nx = distribution.extent(0);
    const int ny = distribution.extent(1);
    const int left_neighbor = domain.left_neighbor;
    const int right_neighbor = domain.right_neighbor;

    Kokkos::View<double**> send_left("send_left_halo", ny, 3);
    Kokkos::View<double**> send_right("send_right_halo", ny, 3);
    Kokkos::View<double**> receive_left("receive_left_halo", ny, 3);
    Kokkos::View<double**> receive_right("receive_right_halo", ny, 3);
 
    Kokkos::parallel_for("pack_x_halos", Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            send_left(j, 0) = distribution(1, j, 3);
            send_left(j, 1) = distribution(1, j, 6);
            send_left(j, 2) = distribution(1, j, 7);
            send_right(j, 0) = distribution(nx - 2, j, 1);
            send_right(j, 1) = distribution(nx - 2, j, 5);
            send_right(j, 2) = distribution(nx - 2, j, 8);
        });
    Kokkos::fence();

    auto send_left_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_left);
    auto send_right_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_right);
    auto receive_left_host = Kokkos::create_mirror_view(receive_left);
    auto receive_right_host = Kokkos::create_mirror_view(receive_right);

    MPI_Sendrecv(send_left_host.data(), ny * 3, MPI_DOUBLE, left_neighbor, 0, receive_right_host.data(), ny * 3, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right_host.data(), ny * 3, MPI_DOUBLE, right_neighbor, 1, receive_left_host.data(), ny * 3, MPI_DOUBLE, left_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (left_neighbor != MPI_PROC_NULL) {
        Kokkos::deep_copy(receive_left, receive_left_host);
    }
    if (right_neighbor != MPI_PROC_NULL) {
        Kokkos::deep_copy(receive_right, receive_right_host);
    }

    Kokkos::parallel_for("unpack_x_halos", Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            if (left_neighbor != MPI_PROC_NULL) {
                distribution(0, j, 1) = receive_left(j, 0);
                distribution(0, j, 5) = receive_left(j, 1);
                distribution(0, j, 8) = receive_left(j, 2);
            }
            if (right_neighbor != MPI_PROC_NULL) {
                distribution(nx - 1, j, 3) = receive_right(j, 0);
                distribution(nx - 1, j, 6) = receive_right(j, 1);
                distribution(nx - 1, j, 7) = receive_right(j, 2);
            }
        });
    Kokkos::fence();
}
