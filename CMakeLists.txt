cmake_minimum_required(VERSION 3.6)
project(DeNoise LANGUAGES C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(De_NOISE_SRC
        src/denoise.c)
include_directories(src/fftw/)
link_directories(src/fftw/)
include_directories(include)
add_executable(denoise_demo ${De_NOISE_SRC})
target_link_libraries(denoise_demo libfftw3f-3 -lm)