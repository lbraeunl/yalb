#include "collision.h"
#include "initialisation.h"
#include "parameters.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(CollisionTest, BasicAssertions)
{
    EXPECT_EQ(7 * 6, 42);
}

TEST(InitializationTest, GaussianBlobIsSmallSymmetricPerturbation)
{
    Kokkos::View<double**> rho("density", X, Y);
    Kokkos::View<double***> u("velocity", X, Y, 2);

    create_gaussian_blob(rho, u);

    const auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            EXPECT_GE(rho_host(i, j), 1.0);
            EXPECT_LE(rho_host(i, j), 1.01);
            EXPECT_DOUBLE_EQ(u_host(i, j, 0), 0.0);
            EXPECT_DOUBLE_EQ(u_host(i, j, 1), 0.0);
            EXPECT_DOUBLE_EQ(rho_host(i, j), rho_host(X - 1 - i, Y - 1 - j));
        }
    }
}

TEST(InitializationTest, UniformWithBumpRaisesOnlyCenterDensity)
{
    Kokkos::View<double**> rho("density", X, Y);
    Kokkos::View<double***> u("velocity", X, Y, 2);

    create_uniform_with_bump(rho, u);

    const auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            const double expected_density = (i == X / 2 && j == Y / 2) ? 1.01 : 1.0;
            EXPECT_DOUBLE_EQ(rho_host(i, j), expected_density);
            EXPECT_DOUBLE_EQ(u_host(i, j, 0), 0.0);
            EXPECT_DOUBLE_EQ(u_host(i, j, 1), 0.0);
        }
    }
}

TEST(InitializationTest, RandomFieldsStayInStableRanges)
{
    Kokkos::View<double**> rho("density", X, Y);
    Kokkos::View<double***> u("velocity", X, Y, 2);

    create_random_fields(rho, u);

    const auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    bool density_varies = false;
    bool velocity_varies = false;

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            EXPECT_GE(rho_host(i, j), 0.99);
            EXPECT_LE(rho_host(i, j), 1.01);
            EXPECT_GE(u_host(i, j, 0), -0.02);
            EXPECT_LE(u_host(i, j, 0), 0.02);
            EXPECT_GE(u_host(i, j, 1), -0.02);
            EXPECT_LE(u_host(i, j, 1), 0.02);

            density_varies = density_varies || rho_host(i, j) != rho_host(0, 0);
            velocity_varies = velocity_varies
                || u_host(i, j, 0) != u_host(0, 0, 0)
                || u_host(i, j, 1) != u_host(0, 0, 1);
        }
    }

    EXPECT_TRUE(density_varies);
    EXPECT_TRUE(velocity_varies);
}

TEST(InitializationTest, SinusoidalCreatesShearWaveAlongY)
{
    Kokkos::View<double**> rho("density", X, Y);
    Kokkos::View<double***> u("velocity", X, Y, 2);

    create_sinusoidal(rho, u);

    const auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    constexpr double epsilon = 0.01;
    constexpr double two_pi = 6.28318530717958647692;

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            EXPECT_DOUBLE_EQ(rho_host(i, j), 1.0);
            EXPECT_NEAR(u_host(i, j, 0), epsilon * std::sin(two_pi * j / Y), 1e-15);
            EXPECT_DOUBLE_EQ(u_host(i, j, 1), 0.0);
        }
    }
}
