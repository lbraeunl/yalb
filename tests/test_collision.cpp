#include "collision.h"
#include "parameters.h"
#include "utils.h"
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
