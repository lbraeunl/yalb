#ifndef PARAMETERS_H
#define PARAMETERS_H

inline constexpr int X = 40;
inline constexpr int Y = 40;
inline constexpr int NUM_STEPS = 50;
inline constexpr double TAU = 0.8;

enum class WallType {
    Periodic,
    BounceBack,
    Moving
};

struct WallProperties {
    WallType type;
    double velocity_x;
    double velocity_y;
};

inline constexpr WallProperties LEFT_WALL{WallType::BounceBack, 0.0, 0.0};
inline constexpr WallProperties RIGHT_WALL{WallType::BounceBack, 0.0, 0.0};
inline constexpr WallProperties DOWN_WALL{WallType::BounceBack, 0.0, 0.0};
inline constexpr WallProperties UP_WALL{WallType::Moving, 0.1, 0.0};

#endif // PARAMETERS_H
