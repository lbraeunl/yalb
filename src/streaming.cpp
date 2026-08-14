#include "streaming.h"
#include "parameters.h"

Kokkos::View<double***> streaming(Kokkos::View<double***> f, Kokkos::View<double**> rho, Kokkos::View<int*[2]> c)
{
    double weight[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    Kokkos::View<double***> f_next("f_next", X, Y, 9);

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
    return f_next;
}