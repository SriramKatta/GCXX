CPMAddPackage(
  NAME Format.cmake
  VERSION 1.8.2
  GITHUB_REPOSITORY TheLartians/Format.cmake
  OPTIONS
      # set to yes skip cmake formatting
      "FORMAT_SKIP_CMAKE NO"
      # set to yes skip clang formatting
      "FORMAT_SKIP_CLANG NO"
      # path to exclude (optional, supports regular expressions)
      # "CMAKE_FORMAT_EXCLUDE cmake/CPM.cmake"
      # extra arguments for cmake_format (optional)
      # "CMAKE_FORMAT_EXTRA_ARGS -c /path/to/cmake-format.{yaml,json,py}"
)