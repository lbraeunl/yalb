#ifndef PARAMETERS_H
#define PARAMETERS_H

inline constexpr int X = 128;
inline constexpr int Y = 128;
inline constexpr int NUM_STEPS = 8000;
inline constexpr double TAU = 0.59;

enum class WallType {
    Periodic,
    Rigid
};

struct WallProperties {
    WallType type;
    double velocity_x;
    double velocity_y;
};

inline constexpr WallProperties LEFT_WALL{WallType::Rigid, 0.0, 0.0};
inline constexpr WallProperties RIGHT_WALL{WallType::Rigid, 0.0, 0.0};
inline constexpr WallProperties DOWN_WALL{WallType::Rigid, 0.0, 0.0};
inline constexpr WallProperties UP_WALL{WallType::Rigid, 0.1, 0.0};

#endif // PARAMETERS_H
