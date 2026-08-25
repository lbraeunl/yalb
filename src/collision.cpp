#include "collision.h"
#include "domain_decomposition.h"
#include "parameters.h"

namespace {

void compute_density_range(const Kokkos::View<double***>& f, const Kokkos::View<double**>& rho, const int x_begin, const int x_end)
{
    const int ny = f.extent_int(1);
    Kokkos::parallel_for("compute_density",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_begin, 0}, {x_end, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double sum = 0.0;
            for (int k = 0; k < 9; ++k) {
                sum += f(i, j, k);
            }
            rho(i, j) = sum;
        });
}


void compute_velocity_range(const Kokkos::View<double***>& f, const Kokkos::View<double**>& rho, const Kokkos::View<int*[2]>& c, const Kokkos::View<double***>& u, const int x_begin, const int x_end)
{
    const int ny = f.extent_int(1);
    Kokkos::parallel_for("compute_velocity",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_begin, 0}, {x_end, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int d = 0; d < 2; ++d) {
                double sum = 0.0;
                for (int k = 0; k < 9; ++k) {
                    sum += f(i, j, k) * c(k, d);
                }
                u(i, j, d) = sum / rho(i, j);
            }
        });
}


void compute_f_eq_range(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u, const Kokkos::View<int*[2]>& c, const Kokkos::View<double***>& f_eq, const int x_begin, const int x_end)
{
    const int ny = rho.extent_int(1);
    Kokkos::parallel_for("compute_f_eq",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_begin, 0}, {x_end, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double ux = u(i, j, 0);
            const double uy = u(i, j, 1);
            const double u_squared = ux * ux + uy * uy;

            for (int k = 0; k < 9; ++k) {
                const double weight = k == 0 ? 4.0 / 9.0
                                    : k < 5 ? 1.0 / 9.0
                                            : 1.0 / 36.0;
                const double c_dot_u = c(k, 0) * ux + c(k, 1) * uy;

                f_eq(i, j, k) = weight * rho(i, j)
                    * (1.0 + 3.0 * c_dot_u
                       + 4.5 * c_dot_u * c_dot_u
                       - 1.5 * u_squared);
            }
        });
}


void compute_f_new_range(const Kokkos::View<double***>& f, const Kokkos::View<double***>& f_eq, const Kokkos::View<double***>& f_post_collision, const int x_begin, const int x_end)
{
    const int ny = f.extent_int(1);
    constexpr double omega = 1.0 / TAU;
    Kokkos::parallel_for("compute_f_new",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({x_begin, 0, 0}, {x_end, ny, 9}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            f_post_collision(i, j, k) = f(i, j, k)
                + omega * (f_eq(i, j, k) - f(i, j, k));
        });
}

} // namespace


void compute_density(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho)
{
    compute_density_range(f, rho, 0, f.extent_int(0));
}


void compute_density(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Domain& domain)
{
    compute_density_range(f, rho, 1, domain.local_nx + 1);
}


void compute_velocity(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& u)
{
    compute_velocity_range(f, rho, c, u, 0, f.extent_int(0));
}


void compute_velocity(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double**>& rho,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& u,
    const Domain& domain)
{
    compute_velocity_range(f, rho, c, u, 1, domain.local_nx + 1);
}


void compute_f_eq(
    const Kokkos::View<double**>& rho,
    const Kokkos::View<double***>& u,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& f_eq)
{
    compute_f_eq_range(rho, u, c, f_eq, 0, rho.extent_int(0));
}


void compute_f_eq(
    const Kokkos::View<double**>& rho,
    const Kokkos::View<double***>& u,
    const Kokkos::View<int*[2]>& c,
    const Kokkos::View<double***>& f_eq,
    const Domain& domain)
{
    compute_f_eq_range(rho, u, c, f_eq, 1, domain.local_nx + 1);
}


void compute_f_new(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double***>& f_eq,
    const Kokkos::View<double***>& f_post_collision)
{
    compute_f_new_range(f, f_eq, f_post_collision, 0, f.extent_int(0));
}


void compute_f_new(
    const Kokkos::View<double***>& f,
    const Kokkos::View<double***>& f_eq,
    const Kokkos::View<double***>& f_post_collision,
    const Domain& domain)
{
    compute_f_new_range(
        f, f_eq, f_post_collision, 1, domain.local_nx + 1);
}
