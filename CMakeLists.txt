CMAKE_MINIMUM_REQUIRED(VERSION 3.0.2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(CMAKE_CXX_COMPILER "clang++")
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wc++11-extensions")

ADD_EXECUTABLE(
  Raytracer2017
  source/main.cpp)

SET_TARGET_PROPERTIES(Raytracer2017 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
)
