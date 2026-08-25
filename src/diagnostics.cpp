#include "diagnostics.h"

#include <stdexcept>

double global_mass(const Kokkos::View<double**>& density)
{
    const int nx = density.extent_int(0);
    const int ny = density.extent_int(1);
    double local_mass = 0.0;
    Kokkos::parallel_reduce("local_mass", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int i, const int j, double& sum) {
            sum += density(i, j);
        },
        local_mass);
    Kokkos::fence();

    double mass = 0.0;
    MPI_Allreduce(&local_mass, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return mass;
}


double global_mass(const Kokkos::View<double**>& density, const Domain& domain)
{
    const int ny = density.extent_int(1);
    double local_mass = 0.0;
    Kokkos::parallel_reduce("local_mass", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0},{domain.local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int i, const int j, double& sum) {
            sum += density(i, j);
        },
        local_mass);
    Kokkos::fence();

    double mass = 0.0;
    MPI_Allreduce(&local_mass, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return mass;
}


double global_kinetic_energy(const Kokkos::View<double**>& density, const Kokkos::View<double***>& velocity)
{
    const int nx = density.extent_int(0);
    const int ny = density.extent_int(1);
    double local_energy = 0.0;
    Kokkos::parallel_reduce("local_kinetic_energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int i, const int j, double& sum) {
            const double ux = velocity(i, j, 0);
            const double uy = velocity(i, j, 1);
            sum += 0.5 * density(i, j) * (ux * ux + uy * uy);
        },
        local_energy);
    Kokkos::fence();

    double energy = 0.0;
    MPI_Allreduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return energy;
}


double global_kinetic_energy(const Kokkos::View<double**>& density, const Kokkos::View<double***>& velocity, const Domain& domain)
{
    const int ny = density.extent_int(1);
    double local_energy = 0.0;
    Kokkos::parallel_reduce("local_kinetic_energy",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0},{domain.local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int i, const int j, double& sum) {
            const double ux = velocity(i, j, 0);
            const double uy = velocity(i, j, 1);
            sum += 0.5 * density(i, j) * (ux * ux + uy * uy);
        },
        local_energy);
    Kokkos::fence();

    double energy = 0.0;
    MPI_Allreduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return energy;
}


double time_to_solution(const double start_time, const double end_time,const MPI_Comm communicator)
{
    const double local_elapsed_seconds = end_time - start_time;

    double elapsed_seconds = 0.0;
    MPI_Allreduce(&local_elapsed_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, communicator);
    return elapsed_seconds;
}

double mlurps(const int global_nx, const int global_ny, const int number_of_steps, const double elapsed_seconds)
{
    const double lattice_updates = static_cast<double>(global_nx) * static_cast<double>(global_ny) * static_cast<double>(number_of_steps);
    return lattice_updates / (1.0e6 * elapsed_seconds);
}
