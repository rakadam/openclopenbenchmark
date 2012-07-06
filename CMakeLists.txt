cmake_minimum_required(VERSION 2.6)
project (openclopenbenchmark)

include_directories(include)

SET(CMAKE_CXX_FLAGS "-O0 -g")

SET(main_SRC
	driver/main.cpp
        driver/ocl_wrapper.cpp
	tmp/test_wrap_main.cpp
)

file(GLOB test_SRC
	tests/memory/*.ocl
	tests/arithmetic/*.ocl
	tests/control/*.ocl
)

EXECUTE_PROCESS(
	COMMAND mkdir -p tmp
)

foreach(socl ${test_SRC})
EXECUTE_PROCESS(
	COMMAND utils/get_name ${socl}
	OUTPUT_VARIABLE name
)
ADD_CUSTOM_COMMAND(
	OUTPUT "tmp/${name}_wrap.cpp"
	COMMAND utils/splitter ${socl} tmp/
	DEPENDS ${socl} utils/splitter
)
SET(main_SRC
    ${main_SRC}
    "tmp/${name}_wrap.cpp"
)
endforeach(socl)

ADD_CUSTOM_COMMAND(
	OUTPUT tmp/test_wrap_main.cpp
	COMMAND utils/wrap_main_gen ${test_SRC} > tmp/test_wrap_main.cpp
	DEPENDS ${test_SRC} utils/wrap_main_gen
)

ADD_EXECUTABLE(opencl_benchmark ${main_SRC})
TARGET_LINK_LIBRARIES(opencl_benchmark OpenCL gsl cblas)
