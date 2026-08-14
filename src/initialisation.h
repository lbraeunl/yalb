#ifndef INITIALISATION_H
#define INITIALISATION_H

#include <Kokkos_Core.hpp>

Kokkos::View<int*[2]> lattice_velocities();

void create_gaussian_blob(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_uniform_rest(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_uniform_with_bump(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_random_fields(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_sinusoidal(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
Kokkos::View<double***> initialize_distribution(const Kokkos::View<int*[2]>& c);

#endif // INITIALISATION_H
