#include "collision.h"
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

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";

    {
        const auto velocities = lattice_velocities();
        Kokkos::View<double**> rho("density", X, Y);
        Kokkos::View<double***> u("velocity", X, Y, 2);

        create_uniform_rest(rho, u);
        auto distribution = compute_f_eq(rho, u, velocities);

        if (rank == 0) {
            write_csv(distribution, rho, u, filename_for_step(0));
        }

        for (int step = 1; step <= NUM_STEPS; ++step) {

            auto f_eq = compute_f_eq(rho, u, velocities);
            distribution = compute_f_new(distribution, f_eq);

            distribution = streaming(distribution, rho, velocities);

            rho = compute_density(distribution);
            u = compute_velocity(distribution, rho, velocities);

            if (rank == 0) {
                //write_csv(distribution, rho, u, filename_for_step(step));
                if (step == NUM_STEPS) {

                    write_csv(distribution, rho, u, filename_for_step(step));
                    std::cout << "Saved results to file.\n";
                }
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
