#include "streaming.h"
#include "domain_decomposition.h"
#include "parameters.h"

void streaming(const Kokkos::View<double***>& f, const Kokkos::View<double**>& rho, const Kokkos::View<int*[2]>& c, const Kokkos::View<double***>& f_next)
{
    constexpr double weight[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    Kokkos::parallel_for("streaming", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double wall_density = rho(i, j);

            for (int k = 0; k < 9; ++k) {
                int ni = i - c(k, 0);
                int nj = j - c(k, 1);
                int nk = k;
                double correction = 0.0;
                bool bounce = false;

                if (ni < 0) {
                    if (LEFT_WALL.type == WallType::Periodic) {
                        ni += X;
                    }
                    else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 1) * LEFT_WALL.velocity_y;
                    }
                }

                else if (ni >= X) {
                    if (RIGHT_WALL.type == WallType::Periodic) {
                        ni -= X;
                    }
                    else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 1) * RIGHT_WALL.velocity_y;
                    }
                }

                if (nj < 0) {
                    if (DOWN_WALL.type == WallType::Periodic) {
                        nj += Y;
                    }
                    else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 0) * DOWN_WALL.velocity_x;
                    }
                }

                else if (nj >= Y) {
                    if (UP_WALL.type == WallType::Periodic) {
                        nj -= Y;
                    }
                    else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 0) * UP_WALL.velocity_x;
                    }
                }

                if (bounce) {
                    ni = i;
                    nj = j;
                    nk = opposite_direction(k);
                }

                f_next(i, j, k) = f(ni, nj, nk) + correction;
            }
        });
}

void streaming(const Kokkos::View<double***>& f, const Kokkos::View<double**>& rho, const Kokkos::View<int*[2]>& c, const Kokkos::View<double***>& f_next, const Domain& domain)
{
    constexpr double weight[9] = {
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
    
    const size_t nx = f.extent(0);
    const size_t ny = f.extent(1);
    const int x_offset = domain.x_offset;

    Kokkos::parallel_for("local_streaming", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0},{nx-1, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double wall_density = rho(i, j);

            for (int k = 0; k < 9; ++k) {
                int ni = i - c(k, 0);
                int nj = j - c(k, 1);
                int nk = k;
                double correction = 0.0;
                bool bounce = false;

                const int global_ni = x_offset + ni - 1;

                if (global_ni < 0 && LEFT_WALL.type == WallType::Rigid) {
                    bounce = true;
                    correction += 6.0 * weight[k] * wall_density * c(k, 1) * LEFT_WALL.velocity_y;
                } else if (global_ni >= X && RIGHT_WALL.type == WallType::Rigid) {
                    bounce = true;
                    correction += 6.0 * weight[k] * wall_density * c(k, 1) * RIGHT_WALL.velocity_y;
                }

                if (nj < 0) {
                    if (DOWN_WALL.type == WallType::Periodic) {
                        nj += ny;
                    } else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 0) * DOWN_WALL.velocity_x;
                    }
                } else if (nj >= ny) {
                    if (UP_WALL.type == WallType::Periodic) {
                        nj -= ny;
                    } else {
                        bounce = true;
                        correction += 6.0 * weight[k] * wall_density * c(k, 0) * UP_WALL.velocity_x;
                    }
                }

                if (bounce) {
                    ni = i;
                    nj = j;
                    nk = opposite_direction(k);
                }

                f_next(i, j, k) = f(ni, nj, nk) + correction;
            }
        });
}
