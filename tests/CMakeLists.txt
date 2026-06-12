cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

function(add_mnist_test test_name test_file)
  add_executable(${test_name} ${test_file})

  target_link_libraries(${test_name}
    PRIVATE
    ml_lib
    matrix_lib
    GTest::gtest_main
  )

  target_include_directories(${test_name}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/matrix/include
    ${CMAKE_SOURCE_DIR}/lib/ml/include
  )

  target_compile_features(${test_name} PRIVATE cxx_std_17)

  add_test(NAME ${test_name} COMMAND ${test_name})

  set_tests_properties(${test_name} PROPERTIES
    TIMEOUT 60
    LABELS "mnist"
  )
endfunction()

add_mnist_test(test_matrix test_matrix.cpp)
add_mnist_test(test_neural_network test_neural_network.cpp)
add_mnist_test(test_mnist_loader test_mnist_loader.cpp)