#ifndef STREAMING_H
#define STREAMING_H

#include <Kokkos_Core.hpp>

Kokkos::View<double***> streaming(Kokkos::View<double***> f, Kokkos::View<int*[2]> c);

#endif // STREAMING_H
