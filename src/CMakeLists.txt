add_library(lib STATIC
    collision.cpp
    diagnostics.cpp
    domain_decomposition.cpp
    initialisation.cpp
    streaming.cpp
    utils.cpp
)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)

if(NOT YALB_LATTICE_SIDE STREQUAL "")
    if(NOT YALB_LATTICE_SIDE MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "YALB_LATTICE_SIDE must be a positive integer")
    endif()
    target_compile_definitions(lib PUBLIC YALB_LATTICE_SIDE=${YALB_LATTICE_SIDE})
endif()

if(NOT YALB_NUM_STEPS STREQUAL "")
    if(NOT YALB_NUM_STEPS MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "YALB_NUM_STEPS must be a positive integer")
    endif()
    target_compile_definitions(lib PUBLIC YALB_NUM_STEPS=${YALB_NUM_STEPS})
endif()

if(YALB_DISABLE_FIELD_OUTPUT)
    target_compile_definitions(lib PUBLIC YALB_DISABLE_FIELD_OUTPUT=1)
endif()
