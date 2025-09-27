set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

set(EXE_NAME stream_wrapper)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE gcxx::gcxx)
set_target_properties(
  ${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON
)
