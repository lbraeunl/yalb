#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <Kokkos_Core.hpp>

class Domain;

std::string filename_for_step(int step);

void print_initial_state(
    int rank,
    int size,
    const char* solver_name,
    double mass,
    double kinetic_energy);

void print_final_state(
    int rank,
    int size,
    double mass,
    double kinetic_energy,
    double elapsed_seconds);

void write_csv(
    const Kokkos::View<double***>& distribution,
    const Kokkos::View<double**>& density,
    const Kokkos::View<double***>& velocity,
    const std::string& filename);


void write_csv(
    const Kokkos::View<double***>& distribution,
    const Kokkos::View<double**>& density,
    const Kokkos::View<double***>& velocity,
    const std::string& filename,
    const Domain& domain);

#endif // UTILS_H
