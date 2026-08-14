#ifndef STREAMING_H
#define STREAMING_H

#include <Kokkos_Core.hpp>

Kokkos::View<double***> streaming(Kokkos::View<double***> f, Kokkos::View<double**> rho, Kokkos::View<int*[2]> c);

KOKKOS_INLINE_FUNCTION
int opposite_direction(int k)
{
    switch (k) {
        case 0: return 0;
        case 1: return 3;
        case 2: return 4;
        case 3: return 1;
        case 4: return 2;
        case 5: return 7;
        case 6: return 8;
        case 7: return 5;
        case 8: return 6;
        default: return -1;
    }
}

#endif // STREAMING_H
