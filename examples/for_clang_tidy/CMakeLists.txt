set(EXE_NAME clang_tidy_header)

add_executable(${EXE_NAME} main.cpp)
target_link_libraries(${EXE_NAME} PRIVATE gpucxx::gpucxx)
