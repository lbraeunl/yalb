#include "streaming.h"
#include "parameters.h"

Kokkos::View<double***> streaming(Kokkos::View<double***> f, Kokkos::View<int*[2]> c)
{
    Kokkos::View<double***> f_next("f_next", X, Y, 9);

    Kokkos::parallel_for("streaming", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int k = 0; k < 9; ++k) {
                int ni = (i - c(k, 0) + X) % X;
                int nj = (j - c(k, 1) + Y) % Y;
                f_next(i, j, k) = f(ni, nj, k);
            }
        });
    return f_next;
}
