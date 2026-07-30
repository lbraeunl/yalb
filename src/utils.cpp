#include "utils.h"
#include "parameters.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <Kokkos_Random.hpp>
#include <stdexcept>



std::string filename_for_step(int step)
{
    const std::filesystem::path filename("out/data/distribution.csv");
    return (filename.parent_path() /
            (filename.stem().string() + "_" + std::to_string(step) + filename.extension().string()))
        .string();
}



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



void create_uniform_with_bump(const Kokkos::View<double**>& rho, const Kokkos::View<double***>& u)
{
    constexpr double background_density = 1.0;
    constexpr double center_density = 1.0;
    constexpr int center_x = X / 2;
    constexpr int center_y = Y / 2;

    Kokkos::parallel_for(
        "initialize_uniform_with_bump",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {X, Y}),
        KOKKOS_LAMBDA(const int i, const int j) {
            rho(i, j) = (i == center_x && j == center_y)
                ? center_density
                : background_density;
            u(i, j, 0) = 1.0;
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



void write_csv(
    const Kokkos::View<double***>& distribution,
    const Kokkos::View<double**>& density,
    const Kokkos::View<double***>& velocity,
    const std::string& filename)
{
    const auto distribution_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), distribution);
    const auto density_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density);
    const auto velocity_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), velocity);

    const std::filesystem::path output_path(filename);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    output << "x_index,y_index,population,distribution_value,density_value,u_x,u_y\n"
           << std::setprecision(17);

    for (std::size_t i = 0; i < distribution.extent(0); ++i) {
        for (std::size_t j = 0; j < distribution.extent(1); ++j) {
            for (std::size_t q = 0; q < distribution.extent(2); ++q) {
                output << i << ','
                       << j << ','
                       << q << ','
                       << distribution_host(i, j, q) << ','
                       << density_host(i, j) << ','
                       << velocity_host(i, j, 0) << ','
                       << velocity_host(i, j, 1) << '\n';
            }
        }
    }
}
