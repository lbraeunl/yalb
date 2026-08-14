#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>


std::string filename_for_step(int step)
{
    const std::filesystem::path filename("out/data/distribution.csv");
    return (filename.parent_path() /
            (filename.stem().string() + "_" + std::to_string(step) + filename.extension().string()))
        .string();
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
