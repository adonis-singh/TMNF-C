cmake_minimum_required(VERSION 3.16)
project(tmnf_physics C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Bit-exactness depends on the FP policy in src/tmnf_fp.h: double intermediates,
# single float round on store. Never let the compiler contract these into FMAs
# or reassociate them. -fno-math-errno only drops the errno store around sqrt
# so it inlines to one instruction; results are unchanged. SLP vectorization
# turns some 3-vector stores into 8/16-byte stores next to scalar loads (and
# the reverse), which stalls store forwarding in the hot vehicle force paths;
# the vector code that pays off is written with intrinsics in gm.h.
add_compile_options(
	-Wall -Wextra -ffp-contract=off -fno-fast-math -fno-math-errno
	-fno-tree-slp-vectorize)

# Tunes for the build host (AVX encodings, scheduling); no FMA because of
# -ffp-contract=off, so results are unchanged. Off by default to keep the
# release binaries portable.
option(TMNF_MARCH_NATIVE "Compile with -march=native" OFF)
if(TMNF_MARCH_NATIVE)
	add_compile_options(-march=native)
endif()

file(GLOB PHYSICS_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
add_library(tmnf_physics STATIC ${PHYSICS_SOURCES})
target_include_directories(tmnf_physics PUBLIC "${CMAKE_SOURCE_DIR}/src")

add_library(
	tmnf_physics_shared SHARED
	${PHYSICS_SOURCES}
	"${CMAKE_SOURCE_DIR}/python/tmnf_env_ffi.c")
set_target_properties(tmnf_physics_shared PROPERTIES OUTPUT_NAME tmnf_physics)
target_include_directories(
	tmnf_physics_shared
	PUBLIC "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/python")
target_link_libraries(tmnf_physics_shared PRIVATE m pthread)

# Python/RL uses the shared library. In release builds, let the compiler
# inline across its translation units instead of routing internal physics
# calls through interposable ELF symbols. Keep the static library unchanged
# so --wrap profiling and incremental native test links remain inexpensive.
option(TMNF_SHARED_LTO "Optimize the release CPU shared library at link time" ON)
if(TMNF_SHARED_LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT TMNF_C_IPO_SUPPORTED OUTPUT TMNF_C_IPO_ERROR LANGUAGES C)
	if(TMNF_C_IPO_SUPPORTED)
		set_property(TARGET tmnf_physics_shared PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
		include(CheckCCompilerFlag)
		check_c_compiler_flag("-fno-semantic-interposition" TMNF_NO_INTERPOSITION_SUPPORTED)
		if(TMNF_NO_INTERPOSITION_SUPPORTED)
			target_compile_options(tmnf_physics_shared PRIVATE
				$<$<CONFIG:Release>:-fno-semantic-interposition>)
		endif()
	else()
		message(STATUS "CPU shared-library LTO unavailable: ${TMNF_C_IPO_ERROR}")
	endif()
endif()

# Loads saved libraries for interleaved, engine-only comparisons.
add_executable(bench_physics EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tools/bench_physics.c")
target_include_directories(bench_physics PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(bench_physics PRIVATE ${CMAKE_DL_LIBS} m)

# The cost profiler interposes on hot functions with --wrap.
add_executable(bench_tick EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tools/bench_tick.c")
target_link_libraries(bench_tick tmnf_physics m pthread)
target_compile_options(bench_tick PRIVATE -O3 -DNDEBUG)
foreach(wrapped
	CHmsZoneDynamic_PhysicsStep2
	CHmsCollisionManager_SZone_PrepareCollisions
	CHmsCollisionManager_SZone_DetectCollisionsCorpus
	CHmsZoneDynamic_ComputeCollisionResponse
	TmnfRace_Step)
	target_link_options(bench_tick PRIVATE "-Wl,--wrap=${wrapped}")
endforeach()

add_executable(
	export_viewer_scene
	"${CMAKE_SOURCE_DIR}/tools/export_viewer_scene.c")
target_link_libraries(export_viewer_scene tmnf_physics m)

# CUDA backend. The physics .c files are compiled a second time
# as CUDA C++ through the stubs in src/cuda/dev/, so the GPU runs the same
# source as the CPU library. Numerics: no FMA contraction, IEEE division and
# square root, no flush-to-zero, no fast math.
option(TMNF_CUDA "Build the optional CUDA backend" OFF)
if(TMNF_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
		set(CMAKE_CUDA_ARCHITECTURES 120)
	endif()
	add_executable(bench_physics_cuda EXCLUDE_FROM_ALL tools/bench_physics_cuda.cu)
	target_include_directories(bench_physics_cuda PRIVATE src src/cuda)
	target_link_libraries(bench_physics_cuda PRIVATE ${CMAKE_DL_LIBS})
	file(GLOB TMNF_CUDA_DEVICE_STUBS CONFIGURE_DEPENDS
		"${CMAKE_SOURCE_DIR}/src/cuda/dev/*.cu")
	add_library(tmnf_cuda STATIC
		${TMNF_CUDA_DEVICE_STUBS}
		"${CMAKE_SOURCE_DIR}/src/cuda/tmnf_cuda_env.cu")
	target_include_directories(tmnf_cuda PUBLIC
		"${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/cuda")
	set_target_properties(tmnf_cuda PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
		POSITION_INDEPENDENT_CODE ON)
	set(TMNF_CUDA_EXACT_FLAGS
		$<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
		$<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true>
		$<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=true>
		$<$<COMPILE_LANGUAGE:CUDA>:-ftz=false>
		$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
	target_compile_options(tmnf_cuda PRIVATE ${TMNF_CUDA_EXACT_FLAGS})
	# Device link-time optimization: the physics is split across one stub
	# per .c file, and without it every GmVec3 helper is an ABI call.
	set_target_properties(tmnf_cuda PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
	# The LTO backend runs inside nvlink and does not inherit the compile-time
	# numerics flags; without these it contracts FMAs and the replays diverge
	# at tick 2. CMAKE_CUDA_FLAGS reaches the device-link step of a static
	# library, target_link_options does not.
	string(APPEND CMAKE_CUDA_FLAGS
		" -Xnvlink -Xnvvm=-fma=0 -Xnvlink -Xptxas=--fmad=false")
	set(TMNF_CUDA_MAXRREG "128" CACHE STRING
		"Register cap for the device LTO link (128: 16 warps/SM, +25% at 65k envs)")
	if(TMNF_CUDA_MAXRREG)
		string(APPEND CMAKE_CUDA_FLAGS
			" -Xnvlink -maxrregcount=${TMNF_CUDA_MAXRREG}")
	endif()
	target_link_libraries(tmnf_cuda PUBLIC tmnf_physics)
	# Every CUDA process reserves TMNF_CUDA_STACK_BYTES of local memory for
	# every thread the card can hold (measured 3.9 GB at 16 KB); more than
	# three at once exhaust the card, whatever -j ctest runs with. Tests
	# labelled cuda take one of these slots (tests/CMakeLists.txt).
	if(NOT DEFINED TMNF_CUDA_TEST_SLOTS)
		set(TMNF_CUDA_TEST_SLOTS 3)
	endif()
	file(WRITE "${CMAKE_BINARY_DIR}/ctest_resources.json"
		"{\"version\":{\"major\":1,\"minor\":0},\"local\":[{\"gpu\":"
		"[{\"id\":\"0\",\"slots\":${TMNF_CUDA_TEST_SLOTS}}]}]}\n")
	set(CTEST_RESOURCE_SPEC_FILE "${CMAKE_BINARY_DIR}/ctest_resources.json")

	add_executable(cuda_libm_hardcases
		"${CMAKE_SOURCE_DIR}/tools/cuda_libm_hardcases.cu")
	target_include_directories(cuda_libm_hardcases PRIVATE
		"${CMAKE_SOURCE_DIR}/src")
	target_compile_options(cuda_libm_hardcases PRIVATE
		${TMNF_CUDA_EXACT_FLAGS}
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
	target_link_libraries(cuda_libm_hardcases PRIVATE gomp)

	add_executable(bench_cuda EXCLUDE_FROM_ALL
		"${CMAKE_SOURCE_DIR}/tools/bench_cuda.cu")
	target_link_libraries(bench_cuda PRIVATE tmnf_cuda m)

	# Python binding of the CUDA env (tmnf_rl.cuda_env): the CPU FFI plus
	# tmnf_cuda_env_ffi.c over the device-resident step, in one shared object
	# so a trainer loads one library for its CUDA training env and its CPU
	# evaluation env. The host physics comes from tmnf_cuda's own objects
	# (nvcc's host pass of each include stub defines that source's
	# functions, position independent); the sources without a device stub
	# are compiled here as PIC, since the static tmnf_physics is not.
	set(TMNF_CUDA_HOST_ONLY_SOURCES)
	foreach(source ${PHYSICS_SOURCES})
		get_filename_component(stem "${source}" NAME_WE)
		if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/cuda/dev/${stem}.cu")
			list(APPEND TMNF_CUDA_HOST_ONLY_SOURCES "${source}")
		endif()
	endforeach()
	add_library(
		tmnf_cuda_shared SHARED
		${TMNF_CUDA_HOST_ONLY_SOURCES}
		"${CMAKE_SOURCE_DIR}/python/tmnf_env_ffi.c"
		"${CMAKE_SOURCE_DIR}/python/tmnf_cuda_env_ffi.c")
	set_target_properties(tmnf_cuda_shared PROPERTIES OUTPUT_NAME tmnf_cuda)
	target_include_directories(
		tmnf_cuda_shared
		PUBLIC "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/cuda"
		"${CMAKE_SOURCE_DIR}/python")
	target_link_libraries(tmnf_cuda_shared PRIVATE tmnf_cuda m pthread)
endif()

# No game data or executable is required for the default source build.
set(TMNF_GAME_MASK "" CACHE FILEPATH "Locally generated 16384-byte physics mask")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(TMNF_GAME_MASK)
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/tools/prepare_game_mask.py" header
        "${TMNF_GAME_MASK}" "${CMAKE_BINARY_DIR}/generated/tmnf_local_game_mask.h"
        RESULT_VARIABLE mask_result)
    if(NOT mask_result EQUAL 0)
        message(FATAL_ERROR "Invalid local game mask")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TMNF_GAME_MASK}")
    foreach(target tmnf_physics tmnf_physics_shared tmnf_cuda tmnf_cuda_shared)
        if(TARGET ${target})
            target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
            target_compile_definitions(${target} PRIVATE TMNF_LOCAL_GAME_MASK=1)
        endif()
    endforeach()
else()
    message(STATUS "Source-only build: game simulation requires TMNF_GAME_MASK (docs/LOCAL_ASSETS.md)")
endif()
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
