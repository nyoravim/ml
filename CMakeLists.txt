cmake_minimum_required(VERSION 3.20)
project(ml LANGUAGES C)

find_package(libnyoravim REQUIRED)

set(LOG_DIR "vendor/log.c/src")
add_library(ml_log STATIC "${LOG_DIR}/log.c")
target_include_directories(ml_log PUBLIC ${LOG_DIR})
target_compile_definitions(ml_log PUBLIC LOG_USE_COLOR)

file(GLOB_RECURSE ML_SRC CONFIGURE_DEPENDS "src/*.c")
add_executable(ml ${ML_SRC})

target_link_libraries(
    ml PUBLIC

    ml_log

    libnyoravim
    z # zlib
    m # math.h
)
