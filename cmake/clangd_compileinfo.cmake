# Tell cmake to generate a json file of compile commands for clangd:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Execute a process without causing a fatal error if it fails
function(gcxx_execute_non_fatal_process)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "COMMAND"
    ${ARGN}
  )
  execute_process(
    COMMAND ${ARG_COMMAND}
    RESULT_VARIABLE result
    OUTPUT_QUIET ERROR_QUIET
  )
endfunction()

# Symlink the compile command output to the source dir, where clangd will find
# it.
set(compile_commands_file "${CMAKE_BINARY_DIR}/compile_commands.json")
set(compile_commands_link "${CMAKE_SOURCE_DIR}/compile_commands.json")
message(
  STATUS
    "Creating symlink from ${compile_commands_link} to ${compile_commands_file}..."
)
gcxx_execute_non_fatal_process(
  COMMAND
  "${CMAKE_COMMAND}"
  -E
  rm
  -f
  "${compile_commands_link}"
)
gcxx_execute_non_fatal_process(
  COMMAND
  "${CMAKE_COMMAND}"
  -E
  touch
  "${compile_commands_file}"
)

gcxx_execute_non_fatal_process(
  COMMAND
  "${CMAKE_COMMAND}"
  -E
  create_symlink
  "${compile_commands_file}"
  "${compile_commands_link}"
)
