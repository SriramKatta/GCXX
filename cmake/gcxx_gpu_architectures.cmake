# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Sriram Katta
#
# Default GPU architectures for GCXX's own examples, tests, and benchmarks.
# Included only when GCXX is the top-level project, so consumer projects are
# never affected.

# NOTE: These architectures reflect the GPUs available to the author for
# testing. The library itself is not limited to these — override
# CMAKE_CUDA_ARCHITECTURES or CMAKE_HIP_ARCHITECTURES in your own project
# (or via -DCMAKE_CUDA_ARCHITECTURES=... on the cmake command line) to
# target any architecture supported by your toolchain.

if(GCXX_CUDA_MODE)
  set(CMAKE_CUDA_ARCHITECTURES
      80 # for A100
      86 # for A40
      90 # for GH200
  )
  # https://developer.nvidia.com/cuda/gpus
elseif(GCXX_HIP_MODE)
  set(CMAKE_HIP_ARCHITECTURES
      gfx908 # for MI100
      gfx90a # for MI210A
      gfx1030 # for RX 6900 XT
      gfx942 # for MI300X & MI300A
  )
  # https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
endif()
