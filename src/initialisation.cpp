#include "initialisation.h"
#include "domain_decomposition.h"
#include "parameters.h"

#include <Kokkos_Random.hpp>


Kokkos::View<int*[2]> lattice_velocities()
{
    Kokkos::View<int*[2]> c("lattice_velocities", 9);

    Kokkos::parallel_for(
        "initialize_lattice_velocities",
        Kokkos::RangePolicy<>(0, 9),
        KOKKOS_LAMBDA(const int q) {
            switch (q) {
            case 0: c(q, 0) =  0; c(q, 1) =  0; break;
            case 1: c(q, 0) =  1; c(q, 1) =  0; break;
            case 2: c(q, 0) =  0; c(q, 1) =  1; break;
            case 3: c(q, 0) = -1; c(q, 1) =  0; break;
            case 4: c(q, 0) =  0; c(q, 1) = -1; break;
            case 5: c(q, 0) =  1; c(q, 1) =  1; break;
            case 6: c(q, 0) = -1; c(q, 1) =  1; break;
            case 7: c(q, 0) = -1; c(q, 1) = -1; break;
            case 8: c(q, 0) =  1; c(q, 1) = -1; break;
            }
        });

    return c;
}


void create_gaussian_blob(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    constexpr double background_density = 1.0;
    constexpr double relative_amplitude = 0.01;

    const double center_x = 0.5 * (X - 1);
    const double center_y = 0.5 * (Y - 1);
    const int smaller_dimension = X < Y ? X : Y;
    const double sigma = smaller_dimension > 1 ? 0.2 * smaller_dimension : 1.0;
    const double inverse_two_sigma_squared = 0.5 / (sigma * sigma);

    Kokkos::parallel_for(
        "initialize_gaussian_blob",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double dx = i - center_x;
            const double dy = j - center_y;
            const double gaussian = Kokkos::exp(
                -(dx * dx + dy * dy) * inverse_two_sigma_squared);
            rho(i, j) = background_density * (1.0 + relative_amplitude * gaussian);
            u(i, j, 0) = 0.0;
            u(i, j, 1) = 0.0;
        });
}


void create_uniform_rest(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    Kokkos::parallel_for(
        "initialize_uniform_rest",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            rho(i, j) = 1.0;
            u(i, j, 0) = 0.0;
            u(i, j, 1) = 0.0;
        });
}


void create_uniform_rest(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u, const Domain& domain)
{
    Kokkos::parallel_for("initialize_local_uniform_rest", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {domain.local_nx + 1, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            rho(i, j) = 1.0;
            u(i, j, 0) = 0.0;
            u(i, j, 1) = 0.0;
        });
}


void create_uniform_with_bump(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    constexpr double background_density = 1.0;
    constexpr double center_density = 1.01;
    constexpr int center_x = X / 2;
    constexpr int center_y = Y / 2;

    Kokkos::parallel_for(
        "initialize_uniform_with_bump",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            rho(i, j) = (i == center_x && j == center_y)
                ? center_density
                : background_density;
            u(i, j, 0) = 0.0;
            u(i, j, 1) = 0.0;
        });
}


void create_random_fields(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    constexpr double minimum_density = 0.99;
    constexpr double maximum_density = 1.01;
    constexpr double maximum_speed_component = 0.02;
    constexpr uint64_t random_seed = 20260714;

    Kokkos::Random_XorShift64_Pool<> random_pool(random_seed);

    Kokkos::parallel_for(
        "initialize_random_fields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            auto generator = random_pool.get_state();

            rho(i, j) = generator.drand(minimum_density, maximum_density);
            u(i, j, 0) = generator.drand(-maximum_speed_component, maximum_speed_component);
            u(i, j, 1) = generator.drand(-maximum_speed_component, maximum_speed_component);

            random_pool.free_state(generator);
        });
}


void create_sinusoidal(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    constexpr double density = 1.0;
    constexpr double epsilon = 0.01;
    constexpr double two_pi = 6.28318530717958647692;

    Kokkos::parallel_for(
        "initialize_sinusoidal_shear_wave",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            rho(i, j) = density;
            u(i, j, 0) = epsilon * Kokkos::sin(two_pi * j / Y);
            u(i, j, 1) = 0.0;
        });
}


Kokkos::View<double***> initialize_distribution(const Kokkos::View<int*[2]>& c)
{
    Kokkos::View<double***> f("westward_blob_distribution", X, Y, 9);
    const double center_x = 0.5 * (X - 1);
    const double center_y = 0.5 * (Y - 1);
    const int smaller_dimension = X < Y ? X : Y;
    const double sigma = smaller_dimension > 1 ? 0.2 * smaller_dimension : 1.0;

    Kokkos::parallel_for(
        "initialize_westward_blob_distribution",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const double dx = i - center_x;
            const double dy = j - center_y;
            const double blob = Kokkos::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));

            for (int q = 0; q < 9; ++q) {
                f(i, j, q) = (c(q, 0) == 1 && c(q, 1) == 1) ? blob : 0.0;
            }
        });

    return f;
}
