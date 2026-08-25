#ifndef PARAMETERS_H
#define PARAMETERS_H

#ifdef YALB_LATTICE_SIDE
inline constexpr int X = YALB_LATTICE_SIDE;
inline constexpr int Y = YALB_LATTICE_SIDE;
#else
inline constexpr int X = 128;
inline constexpr int Y = 128;
#endif

#ifdef YALB_NUM_STEPS
inline constexpr int NUM_STEPS = YALB_NUM_STEPS;
#else
inline constexpr int NUM_STEPS = 8000;
#endif

inline constexpr double TAU = 0.59;

inline constexpr bool USE_DOMAIN_DECOMPOSITION = true;

#ifdef YALB_DISABLE_FIELD_OUTPUT
inline constexpr bool WRITE_FIELD_OUTPUT = false;
#else
inline constexpr bool WRITE_FIELD_OUTPUT = false;
#endif

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
