cmake_minimum_required(VERSION 3.10)

project(sms_cpp)

set(CMAKE_CXX_STANDARD 17)

# must return value if you define non-void function
add_compile_options(-Werror=return-type)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(include)

add_subdirectory(libaa)
include_directories(${libaa_SOURCE_DIR}/third_party/Eigen3.3.7)
include_directories(${libaa_SOURCE_DIR}/third_party/kissfft)

set(SMS_SRC
        src/sms_util.cpp
        src/dft_model.cpp
        src/stft_model.cpp
        )

add_library(sms ${SMS_SRC})

add_subdirectory(test)
add_subdirectory(example)