#include "domain_decomposition.h"
#include "initialisation.h"
#include "streaming.h"

#include <gtest/gtest.h>
#include <mpi.h>

TEST(DomainDecompositionTest, AssignsBalancedContiguousXStrips)
{
    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int global_nx = 2 * size + 1;
    const Domain domain(global_nx, 7, rank, size, false);

    const int base = global_nx / size;
    const int remainder = global_nx % size;
    const int expected_nx = base + (rank < remainder ? 1 : 0);
    const int expected_offset = rank * base + (rank < remainder ? rank : remainder);

    EXPECT_EQ(domain.local_nx, expected_nx);
    EXPECT_EQ(domain.x_offset, expected_offset);
    EXPECT_EQ(domain.left_neighbor, rank == 0 ? MPI_PROC_NULL : rank - 1);
    EXPECT_EQ(domain.right_neighbor, rank == size - 1 ? MPI_PROC_NULL : rank + 1);
}

TEST(DomainDecompositionTest, ExchangesOnlyCrossingPopulations)
{
    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        GTEST_SKIP() << "This test requires at least two MPI processes";
    }

    const Domain domain(2 * size + 1, 5, rank, size, false);
    Kokkos::View<double***> f("halo_test_distribution", domain.local_nx + 2, 5, 9);
    auto initial = Kokkos::create_mirror_view(f);
    for (int i = 0; i < domain.local_nx + 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int q = 0; q < 9; ++q) {
                initial(i, j, q) = i >= 1
                        && i <= domain.local_nx
                    ? 1000.0 * rank + 10.0 * j + q
                    : -1.0;
            }
        }
    }
    Kokkos::deep_copy(f, initial);

    halo_exchange(f, domain);
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    for (int j = 0; j < 5; ++j) {
        if (rank > 0) {
            EXPECT_DOUBLE_EQ(host(0, j, 1), 1000.0 * (rank - 1) + 10.0 * j + 1.0);
            EXPECT_DOUBLE_EQ(host(0, j, 5), 1000.0 * (rank - 1) + 10.0 * j + 5.0);
            EXPECT_DOUBLE_EQ(host(0, j, 8), 1000.0 * (rank - 1) + 10.0 * j + 8.0);
            EXPECT_DOUBLE_EQ(host(0, j, 0), -1.0);
        }
        if (rank + 1 < size) {
            const int right_ghost = domain.local_nx + 1;
            EXPECT_DOUBLE_EQ(host(right_ghost, j, 3), 1000.0 * (rank + 1) + 10.0 * j + 3.0);
            EXPECT_DOUBLE_EQ(host(right_ghost, j, 6), 1000.0 * (rank + 1) + 10.0 * j + 6.0);
            EXPECT_DOUBLE_EQ(host(right_ghost, j, 7), 1000.0 * (rank + 1) + 10.0 * j + 7.0);
            EXPECT_DOUBLE_EQ(host(right_ghost, j, 0), -1.0);
        }
    }
}

TEST(DomainDecompositionTest, StreamsAcrossAnInternalStripBoundary)
{
    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        GTEST_SKIP() << "This test requires at least two MPI processes";
    }

    const int ny = 5;
    const Domain domain(3 * size, ny, rank, size, false);
    const auto c = lattice_velocities();
    Kokkos::View<double***> f("stream_test_distribution", domain.local_nx + 2, ny, 9);
    Kokkos::View<double**> rho("stream_test_density", domain.local_nx + 2, ny);
    auto f_initial = Kokkos::create_mirror_view(f);
    auto rho_initial = Kokkos::create_mirror_view(rho);
    for (int i = 0; i < domain.local_nx + 2; ++i) {
        for (int j = 0; j < ny; ++j) {
            rho_initial(i, j) = 1.0;
            for (int q = 0; q < 9; ++q) {
                f_initial(i, j, q) = i >= 1
                        && i <= domain.local_nx
                    ? 1000.0 * (domain.x_offset + i - 1) + 10.0 * j + q
                    : 0.0;
            }
        }
    }
    Kokkos::deep_copy(f, f_initial);
    Kokkos::deep_copy(rho, rho_initial);

    halo_exchange(f, domain);
    Kokkos::View<double***> f_next(
        "stream_test_next_distribution", domain.local_nx + 2, ny, 9);
    streaming(f, rho, c, f_next, domain);
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_next);
    constexpr int j = 2;
    if (rank > 0) {
        const int source_global_x = domain.x_offset - 1;
        EXPECT_DOUBLE_EQ(host(1, j, 1), 1000.0 * source_global_x + 10.0 * j + 1.0);
    }
    if (rank + 1 < size) {
        const int source_global_x = domain.x_offset + domain.local_nx;
        EXPECT_DOUBLE_EQ(
            host(domain.local_nx, j, 3),
            1000.0 * source_global_x + 10.0 * j + 3.0);
    }
}
