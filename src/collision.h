#ifndef COLLISION_H
#define COLLISION_H

#include <Kokkos_Core.hpp>

Kokkos::View<double**> compute_density(Kokkos::View<double***> f);
Kokkos::View<double***> compute_velocity(Kokkos::View<double***> f, Kokkos::View<double**> density, Kokkos::View<int*[2]> c);
Kokkos::View<double***> compute_f_eq(Kokkos::View<double**> density, Kokkos::View<double***> velocity, Kokkos::View<int*[2]> c);
Kokkos::View<double***> compute_f_new(Kokkos::View<double***> distribution, Kokkos::View<double***> f_eq);

#endif // COLLISION_H
