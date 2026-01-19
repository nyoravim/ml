cmake_minimum_required(VERSION 3.20)
project(ml LANGUAGES C)

find_package(libnyoravim REQUIRED)

file(GLOB_RECURSE ML_SRC CONFIGURE_DEPENDS "src/*.c")
add_executable(ml ${ML_SRC})

target_link_libraries(
    ml PUBLIC

    libnyoravim
    z # zlib
    m # math.h
)
