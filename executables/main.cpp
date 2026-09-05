#include "collision.h"
#include "diagnostics.h"
#include "domain_decomposition.h"
#include "initialisation.h"
#include "parameters.h"
#include "streaming.h"
#include "utils.h"

#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>
#include <utility>

namespace {

void run_single_grid(const int rank, const int size)
{
    const auto c = lattice_velocities();
    Kokkos::View<double**> rho("density", X, Y);
    Kokkos::View<double***> u("velocity", X, Y, 2);
    Kokkos::View<double***> f_eq("equilibrium_distribution", X, Y, 9);
    Kokkos::View<double***> f("distribution", X, Y, 9);
    Kokkos::View<double***> f_post_collision("post_collision_distribution", X, Y, 9);
    Kokkos::View<double***> f_next("next_distribution", X, Y, 9);

    //create_uniform_rest(rho, u);
    create_sinusoidal(rho, u);
    compute_f_eq(rho, u, c, f);

    const double initial_mass = global_mass(rho);
    const double initial_energy = global_kinetic_energy(rho, u);
    print_initial_state(rank, size, "single-grid solver", initial_mass, initial_energy);
    if constexpr (WRITE_FIELD_OUTPUT) {
        write_csv(f, rho, u, filename_for_step(0));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    for (int step = 1; step <= NUM_STEPS; ++step) {
        compute_f_eq(rho, u, c, f_eq);
        compute_f_new(f, f_eq, f_post_collision);
        streaming(f_post_collision, rho, c, f_next);
        std::swap(f, f_next);
        compute_density(f, rho);
        compute_velocity(f, rho, c, u);
    }

    Kokkos::fence();
    const double elapsed_seconds = time_to_solution(start_time, MPI_Wtime());
    const double mass = global_mass(rho);
    const double energy = global_kinetic_energy(rho, u);

    if constexpr (WRITE_FIELD_OUTPUT) {
        write_csv(f, rho, u, filename_for_step(NUM_STEPS));
    }
    print_final_state(rank, size, mass, energy, elapsed_seconds);
}


void run_domain_decomposed(const int rank, const int size)
{
    const Domain domain(X, Y, rank, size, PERIODIC_X);
    const auto c = lattice_velocities();
    const int local_X = domain.local_nx + 2;
    Kokkos::View<double**> rho("density", local_X, Y);
    Kokkos::View<double***> u("velocity", local_X, Y, 2);
    Kokkos::View<double***> f_eq("equilibrium_distribution", local_X, Y, 9);
    Kokkos::View<double***> f("distribution", local_X, Y, 9);
    Kokkos::View<double***> f_post_collision("post_collision_distribution", local_X, Y, 9);
    Kokkos::View<double***> f_next("next_distribution", local_X, Y, 9);
    HaloBuffers halo_buffers(Y);

    create_uniform_rest(rho, u, domain);
    compute_f_eq(rho, u, c, f, domain);

    const double initial_mass = global_mass(rho, domain);
    const double initial_energy = global_kinetic_energy(rho, u, domain);
    print_initial_state(rank, size, "domain-decomposed solver", initial_mass, initial_energy);
    if constexpr (WRITE_FIELD_OUTPUT) {
        write_csv(f, rho, u, filename_for_step(0), domain);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    for (int step = 1; step <= NUM_STEPS; ++step) {
        compute_f_eq(rho, u, c, f_eq, domain);
        compute_f_new(f, f_eq, f_post_collision, domain);
        halo_exchange(f_post_collision, domain, halo_buffers);
        streaming(f_post_collision, rho, c, f_next, domain);
        std::swap(f, f_next);
        compute_density(f, rho, domain);
        compute_velocity(f, rho, c, u, domain);
    }

    Kokkos::fence();
    const double elapsed_seconds = time_to_solution(start_time, MPI_Wtime());
    const double mass = global_mass(rho, domain);
    const double energy = global_kinetic_energy(rho, u, domain);

    if constexpr (WRITE_FIELD_OUTPUT) {
        write_csv(f, rho, u, filename_for_step(NUM_STEPS), domain);
    }
    print_final_state(rank, size, mass, energy, elapsed_seconds);
}

}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Halo exchange: "
                  << (cuda_aware_mpi_is_enabled()
                          ? "CUDA-aware MPI with device buffers\n"
                          : "host-staged MPI\n");
    }

    int exit_code = 0;
    {
        if constexpr (USE_DOMAIN_DECOMPOSITION) {
            run_domain_decomposed(rank, size);
        } else if (size == 1) {
            run_single_grid(rank, size);
        } else {
            if (rank == 0) {
                std::cerr
                    << "Domain decomposition must be enabled when using "
                    << size << " MPI processes.\n";
            }
            exit_code = 1;
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return exit_code;
}
