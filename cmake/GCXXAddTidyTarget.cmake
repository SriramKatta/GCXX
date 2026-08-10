# GCXX clang-tidy integration
# Adapted from NVIDIA CCCL's cmake/CCCLAddTidyTarget.cmake.
#
# Provides two functions:
#
#   gcxx_tidy_init()
#     Call once from the top-level CMakeLists when GCXX_ENABLE_CLANG_TIDY=ON.
#     Finds clang-tidy, creates the aggregate gcxx.tidy target, and writes
#     run_clang_tidy.sh + a clang-compatible compile_commands.json into the
#     binary directory.  The compilation database avoids the project's nvcc
#     compile_commands.json so clang-tidy never sees nvcc-specific flags.
#
#   gcxx_tidy_add_target(TARGET <name> SOURCES <src> ...)
#     Register source files to be linted as part of gcxx.tidy.
#     No-op when GCXX_ENABLE_CLANG_TIDY is OFF, so call sites in lib/tests/examples
#     never need to be guarded.
#
#     Include directories and compile definitions are collected from the gpu_cxx
#     target at configure time, so the build compiler (nvcc, nvc++, etc.) does
#     not need to be clang.
#
#     Header files (.hpp .hxx .hh .h .cuh .inl .ipp) receive -x c++-header.

# Provide no-op stubs when the feature is off so callers need no guards.
if(NOT GCXX_ENABLE_CLANG_TIDY)
  macro(gcxx_tidy_init)

  endmacro()
  macro(gcxx_tidy_add_target)

  endmacro()
  return()
endif()

# ---------------------------------------------------------------------------
# Internal helper — written to file after all gcxx_tidy_add_target() calls.
# ---------------------------------------------------------------------------
function(_gcxx_tidy_write_db)
  get_property(_entries GLOBAL PROPERTY _GCXX_TIDY_DB_ENTRIES)
  get_property(_db_path GLOBAL PROPERTY _GCXX_TIDY_DB_PATH)

  if(_entries)
    list(LENGTH _entries _count)
    list(
      JOIN
      _entries
      ",\n"
      _body
    )
    file(WRITE "${_db_path}" "[\n${_body}\n]\n")
    message(
      STATUS
        "clang-tidy: wrote ${_count}-entry compilation database → ${_db_path}"
    )
  else()
    file(WRITE "${_db_path}" "[]\n")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal helper — append a target's INTERFACE_INCLUDE_DIRECTORIES to _flags.
# ---------------------------------------------------------------------------
function(_gcxx_tidy_add_includes target flag_prefix)
  if(NOT TARGET "${target}")
    return()
  endif()
  get_target_property(_incs "${target}" INTERFACE_INCLUDE_DIRECTORIES)
  if(_incs)
    foreach(_inc IN LISTS _incs)
      list(APPEND _flags "${flag_prefix}${_inc}")
    endforeach()
    set(_flags
        "${_flags}"
        PARENT_SCOPE
    )
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal helper — collects compiler flags from cmake targets.
# Must be called after gpu_cxx and its dependencies are defined.
# ---------------------------------------------------------------------------
function(_gcxx_tidy_collect_flags out_var)
  set(_flags "-std=c++${CMAKE_CXX_STANDARD}" "-Wno-unknown-warning-option"
             "-Wno-error=unused-command-line-argument"
  )

  # Direct include directories on gpu_cxx (e.g. lib/)
  if(TARGET gpu_cxx)
    _gcxx_tidy_add_includes(gpu_cxx "-I")

    # Compile definitions (GCXX_CUDA_MODE, MDSPAN_IMPL_*, …)
    get_target_property(_defs gpu_cxx INTERFACE_COMPILE_DEFINITIONS)
    if(_defs)
      foreach(_def IN LISTS _defs)
        list(APPEND _flags "-D${_def}")
      endforeach()
    endif()
  endif()

  # mdspan — system include so its own warnings are suppressed
  _gcxx_tidy_add_includes(mdspan_headers "-isystem")

  # CUDA runtime + BLAS backend headers — system includes
  if(GCXX_CUDA_MODE)
    _gcxx_tidy_add_includes(CUDA::cudart "-isystem")
    _gcxx_tidy_add_includes(CUDA::cublas "-isystem")
  endif()

  # HIP runtime + BLAS backend headers — system includes
  if(GCXX_HIP_MODE)
    _gcxx_tidy_add_includes(hip::host "-isystem")
    _gcxx_tidy_add_includes(hip::hipblas "-isystem")
  endif()

  # Stub out CUDA qualifiers so clang parses GPU headers without a CUDA
  # compiler.  These are also in .clang-tidy ExtraArgsBefore for belt-and-suspenders.
  list(
    APPEND
    _flags
    "-D__host__="
    "-D__device__="
    "-D__global__="
    "-D__shared__="
    "-D__constant__="
    "-D__managed__="
    "-D__forceinline__=__attribute__((always_inline))"
  )

  set(${out_var}
      "${_flags}"
      PARENT_SCOPE
  )
endfunction()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

function(gcxx_tidy_init)
  find_program(GCXX_CLANG_TIDY clang-tidy REQUIRED)
  message(STATUS "clang-tidy: ${GCXX_CLANG_TIDY}")

  # Global umbrella target
  add_custom_target(
    gcxx.tidy COMMENT "Run clang-tidy on all registered GCXX sources"
  )

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_clang_tidy.sh.in"
    "${CMAKE_BINARY_DIR}/run_clang_tidy.sh" @ONLY
  )
  file(
    CHMOD
    "${CMAKE_BINARY_DIR}/run_clang_tidy.sh"
    PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
  )

  # Prepare a clang-compatible compilation database in tidy_db/.
  # Entries are collected by gcxx_tidy_add_target() and written at the end of
  # configure via cmake_language(DEFER …).  This database is used by
  # run_clang_tidy.sh so clang-tidy never sees the nvcc compile_commands.json.
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tidy_db")
  set_property(GLOBAL PROPERTY _GCXX_TIDY_DB_ENTRIES "")
  set_property(
    GLOBAL PROPERTY _GCXX_TIDY_DB_PATH
                    "${CMAKE_BINARY_DIR}/tidy_db/compile_commands.json"
  )
  cmake_language(
    DEFER
    DIRECTORY
    "${CMAKE_SOURCE_DIR}"
    CALL
    _gcxx_tidy_write_db
  )
endfunction()

function(gcxx_tidy_add_target)
  cmake_parse_arguments(
    ARG
    ""
    "TARGET"
    "SOURCES"
    ${ARGN}
  )

  if(NOT ARG_TARGET)
    message(FATAL_ERROR "gcxx_tidy_add_target: TARGET is required")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gcxx_tidy_add_target: SOURCES is required")
  endif()
  if(NOT TARGET gcxx.tidy)
    return() # gcxx_tidy_init() was not called (e.g. used as a subdirectory)
  endif()

  _gcxx_tidy_collect_flags(_flags)

  # Build the JSON "arguments" array fragment shared by all entries from this call.
  # Each flag is double-quoted for JSON; backslashes and quotes inside are escaped.
  set(_json_flags "\"${GCXX_CLANG_TIDY}\"")
  foreach(_flag IN LISTS _flags)
    string(
      REPLACE "\\"
              "\\\\"
              _f
              "${_flag}"
    )
    string(
      REPLACE "\""
              "\\\""
              _f
              "${_f}"
    )
    string(APPEND _json_flags ", \"${_f}\"")
  endforeach()

  set(_header_exts
      .hpp
      .hxx
      .hh
      .h
      .cuh
      .inl
      .ipp
  )

  foreach(_src IN LISTS ARG_SOURCES)
    get_filename_component(
      _abs
      "${_src}"
      ABSOLUTE
      BASE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    file(
      RELATIVE_PATH
      _rel
      "${CMAKE_SOURCE_DIR}"
      "${_abs}"
    )

    # Stable cmake target name
    string(
      REPLACE "/"
              "."
              _slug
              "${_rel}"
    )
    string(
      REPLACE ".."
              "__"
              _slug
              "${_slug}"
    )
    set(_tidy_tgt "tidy.${_slug}")

    # Header files need explicit language forcing
    get_filename_component(_ext "${_src}" EXT)
    set(_lang_json "")
    if(_ext IN_LIST _header_exts)
      set(_lang_json "\"-x\", \"c++-header\", ")
    endif()

    # Register a clang-compatible compilation database entry for this file.
    # cmake_language(DEFER …) will write all entries to tidy_db/compile_commands.json
    # at the end of configure so clang-tidy finds a proper database.
    set(_entry
        "  {\n    \"directory\": \"${CMAKE_SOURCE_DIR}\",\n    \"arguments\": [${_json_flags}, ${_lang_json}\"${_abs}\"],\n    \"file\": \"${_abs}\"\n  }"
    )
    set_property(GLOBAL APPEND PROPERTY _GCXX_TIDY_DB_ENTRIES "${_entry}")

    add_custom_target(
      ${_tidy_tgt}
      COMMAND "${CMAKE_BINARY_DIR}/run_clang_tidy.sh" "${_abs}"
      COMMENT "clang-tidy: ${_rel}"
      VERBATIM
    )

    add_dependencies(gcxx.tidy ${_tidy_tgt})
  endforeach()
endfunction()
