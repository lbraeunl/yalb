find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

set(TEST_SOURCES
    test_collision.cpp
    test_domain_decomposition.cpp
)

list(APPEND TEST_SOURCES gtest_mpi_main.cpp)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE lib GTest::gtest)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


include(GoogleTest)
gtest_discover_tests(tests)

if(MPIEXEC_EXECUTABLE)
    add_test(
        NAME domain_decomposition_mpi_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:tests>
                --gtest_filter=DomainDecompositionTest.*
    )
    set_tests_properties(domain_decomposition_mpi_2 PROPERTIES PROCESSORS 2)
endif()
