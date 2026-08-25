#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "domain_decomposition.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

double global_mass(const Kokkos::View<double**>& density);
double global_mass(const Kokkos::View<double**>& density, const Domain& domain);

double global_kinetic_energy(const Kokkos::View<double**>& density, const Kokkos::View<double***>& velocity);
double global_kinetic_energy(const Kokkos::View<double**>& density, const Kokkos::View<double***>& velocity, const Domain& domain);

double time_to_solution(double start_time, double end_time, MPI_Comm communicator = MPI_COMM_WORLD);

double mlurps(int global_nx, int global_ny, int number_of_steps, double elapsed_seconds);

#endif // DIAGNOSTICS_H
