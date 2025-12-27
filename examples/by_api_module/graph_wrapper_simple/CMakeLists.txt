set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GCXX_GPU_LANG})

set(EXE_NAME graph_wrapper_simple
)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE gcxx::gcxx)
target_link_libraries(${EXE_NAME} PRIVATE fmt::fmt)

