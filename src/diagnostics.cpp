// #include "diagnostics.h"

// #include <stdexcept>

// double global_mass(
//     const Kokkos::View<double**>& density,
//     const DomainDecomposition& domain)
// {
//     if (density.extent_int(0) != domain.ghosted_nx()
//         || density.extent_int(1) != domain.global_ny()) {
//         throw std::invalid_argument("Density dimensions do not match the domain");
//     }

//     double local_mass = 0.0;
//     Kokkos::parallel_reduce(
//         "local_mass",
//         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
//             {domain.interior_begin(), 0},
//             {domain.interior_end(), domain.global_ny()}),
//         KOKKOS_LAMBDA(const int i, const int j, double& sum) {
//             sum += density(i, j);
//         },
//         local_mass);
//     Kokkos::fence();

//     double mass = 0.0;
//     MPI_Allreduce(
//         &local_mass, &mass, 1, MPI_DOUBLE, MPI_SUM, domain.communicator());
//     return mass;
// }

// double global_kinetic_energy(
//     const Kokkos::View<double**>& density,
//     const Kokkos::View<double***>& velocity,
//     const DomainDecomposition& domain)
// {
//     if (density.extent_int(0) != domain.ghosted_nx()
//         || density.extent_int(1) != domain.global_ny()
//         || velocity.extent_int(0) != domain.ghosted_nx()
//         || velocity.extent_int(1) != domain.global_ny()
//         || velocity.extent_int(2) != 2) {
//         throw std::invalid_argument("Macroscopic field dimensions do not match the domain");
//     }

//     double local_energy = 0.0;
//     Kokkos::parallel_reduce(
//         "local_kinetic_energy",
//         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
//             {domain.interior_begin(), 0},
//             {domain.interior_end(), domain.global_ny()}),
//         KOKKOS_LAMBDA(const int i, const int j, double& sum) {
//             const double ux = velocity(i, j, 0);
//             const double uy = velocity(i, j, 1);
//             sum += 0.5 * density(i, j) * (ux * ux + uy * uy);
//         },
//         local_energy);
//     Kokkos::fence();

//     double energy = 0.0;
//     MPI_Allreduce(
//         &local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, domain.communicator());
//     return energy;
// }
