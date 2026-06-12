cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

function(add_matrix_test test_name test_file)
  add_executable(${test_name} ${test_file})
    
  target_link_libraries(${test_name}
    PRIVATE
    matrix_lib
    GTest::gtest_main
  )
    
  target_include_directories(${test_name}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/matrix/include
  )
    
  target_compile_features(${test_name} PRIVATE cxx_std_17)
    
  add_test(NAME ${test_name} COMMAND ${test_name})
    
  set_tests_properties(${test_name} PROPERTIES
    TIMEOUT 30
    LABELS "matrix"
  )
endfunction()

add_matrix_test(test_matrix test_matrix.cpp)