set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${BUILD_GPU_LANGUAGE})

set(EXE_NAME stream_wrapper)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE gpucxx::gpucxx)
target_link_libraries(${EXE_NAME} PRIVATE fmt_system)
set_target_properties(${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)
