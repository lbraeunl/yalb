#ifndef COLLISION_H
#define COLLISION_H

#include <Kokkos_Core.hpp>

class Domain;

void compute_density(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho);
void compute_density(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Domain& domain);

void compute_velocity(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& u);
void compute_velocity(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& u,
    const Domain& domain);

void compute_f_eq(
    const Kokkos::View<double**>& rho,
    const Kokkos::View<double***>& u,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& f_eq);
void compute_f_eq(
    const Kokkos::View<double**>& rho,
    const Kokkos::View<double***>& u,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& f_eq,
    const Domain& domain);

void compute_f_new(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double***>& f_eq,
    const Kokkos::View<double***>& f_post_collision);
void compute_f_new(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double***>& f_eq,
    const Kokkos::View<double***>& f_post_collision,
    const Domain& domain);

#endif // COLLISION_H
