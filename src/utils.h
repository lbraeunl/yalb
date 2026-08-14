#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <Kokkos_Core.hpp>

std::string filename_for_step(int step);

void write_csv(
    const Kokkos::View<double***>& distribution,
    const Kokkos::View<double**>& density,
    const Kokkos::View<double***>& velocity,
    const std::string& filename);

#endif // UTILS_H
