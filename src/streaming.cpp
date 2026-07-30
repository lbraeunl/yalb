#include "streaming.h"
#include "parameters.h"

Kokkos::View<double***> streaming(Kokkos::View<double***> f, Kokkos::View<int*[2]> c)
{
    Kokkos::View<double***> f_next("f_next", X, Y, 9);

    Kokkos::parallel_for("streaming", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if (i == 0) {
                // TODO: Apply LEFT_WALL.
            }
            if (i == X - 1) {
                // TODO: Apply RIGHT_WALL.
            }
            if (j == 0) {
                // TODO: Apply DOWN_WALL.
            }
            if (j == Y - 1) {
                // TODO: Apply UP_WALL.
            }

            for (int k = 0; k < 9; ++k) {
                int ni = (i - c(k, 0) + X) % X;
                int nj = (j - c(k, 1) + Y) % Y;
                f_next(i, j, k) = f(ni, nj, k);
            }
        });
    return f_next;
}
