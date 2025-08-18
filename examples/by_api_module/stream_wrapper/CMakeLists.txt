set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${BUILD_GPU_LANGUAGE})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE gpucxx::gpucxx)
target_link_libraries(main PRIVATE fmt::fmt)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)
