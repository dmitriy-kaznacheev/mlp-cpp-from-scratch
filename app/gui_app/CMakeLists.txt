cmake_minimum_required(VERSION 3.16)

add_executable(mnist_gui
  main.cpp
  mainwindow.cpp
  drawingcanvas.cpp
)

target_link_libraries(mnist_gui
  PRIVATE
  ml_lib
  matrix_lib
  Qt::Widgets
)

target_include_directories(mnist_gui
  PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/matrix/include
  ${CMAKE_SOURCE_DIR}/lib/ml/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(mnist_gui PRIVATE cxx_std_17)

set_target_properties(mnist_gui 
  PROPERTIES
  AUTOMOC ON
  AUTOUIC ON
)