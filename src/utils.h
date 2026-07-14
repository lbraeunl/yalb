#ifndef UTILS_H
#define UTILS_H

#include <Kokkos_Core.hpp>
#include <string>

Kokkos::View<int*[2]> lattice_velocities();

void create_gaussian_blob(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_uniform_with_bump(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
void create_random_fields(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u);
Kokkos::View<double***> initialize_distribution(const Kokkos::View<int*[2]>& c);

std::string filename_for_step(int step);

void write_csv(const Kokkos::View<double***>& distribution, const Kokkos::View<double**>& density, const std::string& filename);

#endif // UTILS_H
