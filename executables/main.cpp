#include "collision.h"
#include "diagnostics.h"
#include "domain_decomposition.h"
#include "initialisation.h"
#include "parameters.h"
#include "streaming.h"
#include "utils.h"
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        const Domain domain(X, Y, rank, size, PERIODIC_X);
        const auto velocities = lattice_velocities();

        Kokkos::View<double**> rho("density", domain.local_nx + 2, Y);
        Kokkos::View<double***> u("velocity", domain.local_nx + 2, Y, 2);

        create_uniform_rest(rho, u, domain);
        auto distribution = compute_f_eq(rho, u, velocities);

        // const double initial_mass = global_mass(rho, domain);
        // const double initial_energy = global_kinetic_energy(rho, u, domain);

        if (rank == 0) {
            std::cout << "Running " << size << " MPI process(es) \n";
            // std::cout << "Initial mass: " << initial_mass << ", kinetic energy: " << initial_energy << '\n';
        }
        write_csv_distributed(distribution, rho, u, domain, filename_for_step(0));

        for (int step = 1; step <= NUM_STEPS; ++step) {

            auto f_eq = compute_f_eq(rho, u, velocities);
            distribution = compute_f_new(distribution, f_eq);

            halo_exchange(distribution, domain);
            distribution = streaming(distribution, rho, velocities, domain);

            rho = compute_density(distribution);
            u = compute_velocity(distribution, rho, velocities);

            if (step == NUM_STEPS) {
                write_csv_distributed(distribution, rho, u, domain, filename_for_step(step));

                // const double mass = global_mass(rho, domain);
                // const double energy = global_kinetic_energy(rho, u, domain);

                if (rank == 0) {
                    std::cout << "Saved results to file.\n";
                    // std::cout << "Final mass: " << mass << ", kinetic energy: " << energy << '\n';
                }
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
