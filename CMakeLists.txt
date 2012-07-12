cmake_minimum_required(VERSION 2.6)
project (openclopenbenchmark)

include_directories(include)

SET(CMAKE_CXX_FLAGS "-O0 -g")

SET(main_SRC
	${openclopenbenchmark_SOURCE_DIR}/driver/main.cpp
        ${openclopenbenchmark_SOURCE_DIR}/driver/ocl_wrapper.cpp
	${openclopenbenchmark_BINARY_DIR}/tmp/test_wrap_main.cpp
)

file(GLOB test_SRC
	${openclopenbenchmark_SOURCE_DIR}/tests/memory/*.ocl
	${openclopenbenchmark_SOURCE_DIR}/tests/arithmetic/*.ocl
	${openclopenbenchmark_SOURCE_DIR}/tests/control/*.ocl
        ${openclopenbenchmark_SOURCE_DIR}/tests/image/*.ocl
)

EXECUTE_PROCESS(
	COMMAND mkdir -p ${openclopenbenchmark_BINARY_DIR}/tmp
)

foreach(socl ${test_SRC})
EXECUTE_PROCESS(
	COMMAND ${openclopenbenchmark_SOURCE_DIR}/utils/get_name ${socl}
	OUTPUT_VARIABLE name
)
ADD_CUSTOM_COMMAND(
	OUTPUT "${openclopenbenchmark_BINARY_DIR}/tmp/${name}_wrap.cpp"
	COMMAND ${openclopenbenchmark_SOURCE_DIR}/utils/splitter ${socl} ${openclopenbenchmark_BINARY_DIR}/tmp/
	DEPENDS ${socl} ${openclopenbenchmark_SOURCE_DIR}/utils/splitter
)
SET(main_SRC
    ${main_SRC}
    "${openclopenbenchmark_BINARY_DIR}/tmp/${name}_wrap.cpp"
)
endforeach(socl)

ADD_CUSTOM_COMMAND(
	OUTPUT ${openclopenbenchmark_BINARY_DIR}/tmp/test_wrap_main.cpp
	COMMAND ${openclopenbenchmark_SOURCE_DIR}/utils/wrap_main_gen ${test_SRC} > ${openclopenbenchmark_BINARY_DIR}/tmp/test_wrap_main.cpp
	DEPENDS ${test_SRC} utils/wrap_main_gen
)

ADD_EXECUTABLE(opencl_benchmark ${main_SRC})
TARGET_LINK_LIBRARIES(opencl_benchmark OpenCL gsl cblas)
