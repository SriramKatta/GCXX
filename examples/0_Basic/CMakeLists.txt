set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE gpucxx::gpucxx)
target_link_libraries(main PRIVATE fmt::fmt)