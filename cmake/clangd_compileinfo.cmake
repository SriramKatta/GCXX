# Tell cmake to generate a json file of compile commands for clangd:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink the compile command output to the source dir, where clangd will find
# it.
set(compile_commands_file "${CMAKE_BINARY_DIR}/compile_commands.json")
set(compile_commands_link "${CMAKE_SOURCE_DIR}/compile_commands.json")
message(
  STATUS
    "Creating symlink from ${compile_commands_link} to ${compile_commands_file}..."
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${compile_commands_link}" OUTPUT_QUIET
                                                                 ERROR_QUIET
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E touch "${compile_commands_file}" OUTPUT_QUIET
                                                                 ERROR_QUIET
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E create_symlink "${compile_commands_file}"
          "${compile_commands_link}" OUTPUT_QUIET ERROR_QUIET
)
