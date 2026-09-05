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

if(YALB_ENABLE_CUDA_AWARE_MPI)
    target_compile_definitions(lib PUBLIC YALB_ENABLE_CUDA_AWARE_MPI=1)
    message(STATUS "YALB halo exchange: CUDA-aware MPI enabled")
else()
    message(STATUS "YALB halo exchange: host-staged MPI")
endif()
