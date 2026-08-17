#include "utils.h"
#include "domain_decomposition.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>


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


void write_csv_distributed(
    const Kokkos::View<double***>& distribution,
    const Kokkos::View<double**>& density,
    const Kokkos::View<double***>& velocity,
    const Domain& domain,
    const std::string& filename)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto distribution_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), distribution);
    const auto density_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density);
    const auto velocity_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), velocity);

    const int ny = distribution.extent_int(1);
    const int local_nodes = domain.local_nx * ny;
    std::vector<double> local_distribution(local_nodes * 9);
    std::vector<double> local_density(local_nodes);
    std::vector<double> local_velocity(local_nodes * 2);

    for (int local_i = 1; local_i <= domain.local_nx; ++local_i) {
        for (int j = 0; j < ny; ++j) {
            const int node = (local_i - 1) * ny + j;
            local_density[node] = density_host(local_i, j);
            local_velocity[node * 2] = velocity_host(local_i, j, 0);
            local_velocity[node * 2 + 1] = velocity_host(local_i, j, 1);
            for (int q = 0; q < 9; ++q) {
                local_distribution[node * 9 + q] =
                    distribution_host(local_i, j, q);
            }
        }
    }

    std::vector<int> node_counts;
    std::vector<int> node_offsets;
    std::vector<double> global_distribution;
    std::vector<double> global_density;
    std::vector<double> global_velocity;
    std::vector<int> rank_local_nx;
    std::vector<int> rank_x_offsets;
    if (rank == 0) {
        rank_local_nx.resize(size);
        rank_x_offsets.resize(size);
        node_counts.resize(size);
        node_offsets.resize(size);
    }

    MPI_Gather(
        &domain.local_nx, 1, MPI_INT,
        rank_local_nx.data(), 1, MPI_INT,
        0, MPI_COMM_WORLD);
    MPI_Gather(
        &domain.x_offset, 1, MPI_INT,
        rank_x_offsets.data(), 1, MPI_INT,
        0, MPI_COMM_WORLD);

    int global_nx = 0;
    MPI_Allreduce(
        &domain.local_nx, &global_nx, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int process = 0; process < size; ++process) {
            node_counts[process] = rank_local_nx[process] * ny;
            node_offsets[process] = rank_x_offsets[process] * ny;
        }
        const int global_nodes = global_nx * ny;
        global_distribution.resize(global_nodes * 9);
        global_density.resize(global_nodes);
        global_velocity.resize(global_nodes * 2);
    }

    auto scaled = [](const std::vector<int>& values, int factor) {
        std::vector<int> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[i] = values[i] * factor;
        }
        return result;
    };
    const auto distribution_counts = scaled(node_counts, 9);
    const auto distribution_offsets = scaled(node_offsets, 9);
    const auto velocity_counts = scaled(node_counts, 2);
    const auto velocity_offsets = scaled(node_offsets, 2);

    MPI_Gatherv(
        local_distribution.data(), local_nodes * 9, MPI_DOUBLE,
        global_distribution.data(), distribution_counts.data(),
        distribution_offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(
        local_density.data(), local_nodes, MPI_DOUBLE,
        global_density.data(), node_counts.data(), node_offsets.data(),
        MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(
        local_velocity.data(), local_nodes * 2, MPI_DOUBLE,
        global_velocity.data(), velocity_counts.data(), velocity_offsets.data(),
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }

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
    for (int i = 0; i < global_nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int node = i * ny + j;
            for (int q = 0; q < 9; ++q) {
                output << i << ',' << j << ',' << q << ','
                       << global_distribution[node * 9 + q] << ','
                       << global_density[node] << ','
                       << global_velocity[node * 2] << ','
                       << global_velocity[node * 2 + 1] << '\n';
            }
        }
    }
}
