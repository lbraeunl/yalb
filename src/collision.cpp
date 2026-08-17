#include "collision.h"
#include "parameters.h"

Kokkos::View<double**> compute_density(Kokkos::View<double***> f)
{
    const int nx = f.extent_int(0);
    const int ny = f.extent_int(1);

    Kokkos::View<double**> density("density", nx, ny);

    Kokkos::parallel_for("compute_density", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {nx - 1, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double sum = 0.0;
            for (int k = 0; k < 9; ++k) {
                sum += f(i, j, k);
            }
            density(i, j) = sum;
        });

    return density;
}


Kokkos::View<double***> compute_velocity(Kokkos::View<double***> f, Kokkos::View<double**> density, Kokkos::View<int*[2]> c)
{
    const int nx = f.extent_int(0);
    const int ny = f.extent_int(1);
   
    Kokkos::View<double***> u("velocity", nx, ny, 2);

    Kokkos::parallel_for("compute_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {nx - 1, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int d = 0; d < 2; ++d) {
                double sum = 0.0;
                for (int k = 0; k < 9; ++k) {
                    sum += f(i, j, k) * c(k, d);
                }
                u(i, j, d) = sum / density(i, j);
            }
        });

    return u;
}


Kokkos::View<double***> compute_f_eq(Kokkos::View<double**> density, Kokkos::View<double***> velocity, Kokkos::View<int*[2]> c)
{
    const int nx = density.extent_int(0);
    const int ny = density.extent_int(1);

    Kokkos::View<double***> f_eq("equilibrium_distribution", nx, ny, 9);

    Kokkos::parallel_for("compute_f_eq", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {nx - 1, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double ux = velocity(i, j, 0);
            const double uy = velocity(i, j, 1);
            const double u_squared = ux * ux + uy * uy;

            for (int k = 0; k < 9; ++k) {
                const double weight = k == 0 ? 4.0 / 9.0
                                    : k < 5 ? 1.0 / 9.0
                                            : 1.0 / 36.0;
                const double c_dot_u = c(k, 0) * ux + c(k, 1) * uy;

                f_eq(i, j, k) = weight * density(i, j) * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u_squared);
            }
        });

    return f_eq;
}


Kokkos::View<double***> compute_f_new(Kokkos::View<double***> distribution, Kokkos::View<double***> f_eq)
{
    const int nx = distribution.extent_int(0);
    const int ny = distribution.extent_int(1);

    Kokkos::View<double***> f_new("post_collision_distribution", nx, ny, 9);
    constexpr double omega = 1.0 / TAU;

    Kokkos::parallel_for("compute_f_new", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {nx - 1, ny, 9}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            f_new(i, j, k) = distribution(i, j, k) + omega * (f_eq(i, j, k) - distribution(i, j, k));
        });

    return f_new;
}
