# Synthetic checks are independent of a game installation.
foreach(name physics_smoke vehicle_compute_smoke gate_observation ellipsoid_vertex_rounding collision_merge_rounding)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE tmnf_physics m)
    target_compile_options(${name} PRIVATE -UNDEBUG)
    add_test(NAME ${name} COMMAND ${name})
endforeach()
# Exercise the device's scalar geometry branches with an optimizing C
# compiler too, so aliasing defects cannot hide behind the host SSE path.
add_executable(ellipsoid_vertex_rounding_scalar ellipsoid_vertex_rounding.c
    "${CMAKE_SOURCE_DIR}/src/collision.c")
target_link_libraries(ellipsoid_vertex_rounding_scalar PRIVATE tmnf_physics m)
target_compile_options(ellipsoid_vertex_rounding_scalar PRIVATE -UNDEBUG -U__SSE2__)
add_test(NAME ellipsoid_vertex_rounding_scalar COMMAND ellipsoid_vertex_rounding_scalar)
add_executable(race_env race_env.c)
target_link_libraries(race_env PRIVATE tmnf_physics m pthread)
target_compile_options(race_env PRIVATE -UNDEBUG)
add_test(NAME race_rules COMMAND race_env --rules-only)
add_executable(track_smoke track_smoke.c)
target_link_libraries(track_smoke PRIVATE tmnf_physics m)
set(fixture "${CMAKE_CURRENT_BINARY_DIR}/synthetic.tmnftrack")
add_test(NAME synthetic_track_generate COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tools/make_track_fixture.py" "${fixture}")
set_tests_properties(synthetic_track_generate PROPERTIES FIXTURES_SETUP synthetic_track)
add_test(NAME synthetic_track_load COMMAND track_smoke "${fixture}")
set_tests_properties(synthetic_track_load PROPERTIES FIXTURES_REQUIRED synthetic_track)
# Build the capture harness without requiring captures at configure time.
add_executable(replay_tick replay_tick.c)
target_link_libraries(replay_tick PRIVATE tmnf_physics m)
if(TMNF_CUDA)
    add_executable(collection_parity collection_parity.c)
    target_link_libraries(collection_parity PRIVATE tmnf_cuda_shared)
    add_executable(cuda_lockstep cuda_lockstep.c)
    target_link_libraries(cuda_lockstep PRIVATE tmnf_cuda tmnf_physics m)
    # Explicit GPU check, not an automatic ctest dependency on GPU hardware.
    # tmnf_cuda resolves/internalizes its device functions for its own kernel.
    # Compile this small rule kernel's dependency closure into its own link.
    add_executable(checkpoint_set_cuda checkpoint_set_cuda.cu
        "${CMAKE_SOURCE_DIR}/src/cuda/dev/race.cu"
        "${CMAKE_SOURCE_DIR}/src/cuda/dev/route.cu"
        "${CMAKE_SOURCE_DIR}/src/cuda/dev/collision.cu"
        "${CMAKE_SOURCE_DIR}/src/cuda/dev/gm.cu"
        "${CMAKE_SOURCE_DIR}/src/cuda/dev/fastbuffer.cu")
    target_link_libraries(checkpoint_set_cuda PRIVATE tmnf_physics m)
    target_compile_options(checkpoint_set_cuda PRIVATE ${TMNF_CUDA_EXACT_FLAGS})
    set_target_properties(checkpoint_set_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON INTERPROCEDURAL_OPTIMIZATION ON)
    add_executable(replay_tick_cuda replay_tick.c)
    target_compile_definitions(replay_tick_cuda PRIVATE TMNF_REPLAY_CUDA=1)
    target_link_libraries(replay_tick_cuda PRIVATE tmnf_cuda tmnf_physics m)
endif()
add_test(NAME local_asset_tools COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_local_assets.py")
add_test(NAME offline_fixture_tools COMMAND "${Python3_EXECUTABLE}"
    -m unittest discover -s "${CMAKE_SOURCE_DIR}/tools/build_track" -p "test_*.py")
add_test(NAME replay_input_tools COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_wr_replay_inputs.py")
add_executable(wr_passthrough "${CMAKE_SOURCE_DIR}/tools/wr_passthrough.c")
target_link_libraries(wr_passthrough PRIVATE tmnf_physics m pthread)
add_executable(vec_env_matches_replay vec_env_matches_replay.c)
target_link_libraries(vec_env_matches_replay PRIVATE tmnf_physics m pthread)
