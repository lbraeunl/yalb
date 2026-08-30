#ifndef PARAMETERS_H
#define PARAMETERS_H


inline constexpr int X = 6000;
inline constexpr int Y = 6000;

inline constexpr int NUM_STEPS = 2000;

inline constexpr double TAU = 0.59;

inline constexpr bool USE_DOMAIN_DECOMPOSITION = true;

inline constexpr bool WRITE_FIELD_OUTPUT = false;

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

inline constexpr bool PERIODIC_X = (LEFT_WALL.type == WallType::Periodic && RIGHT_WALL.type == WallType::Periodic);

#endif // PARAMETERS_H
