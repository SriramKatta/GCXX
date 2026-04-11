# GCXX

A lightweight, backend-agnostic C++ GPU runtime abstraction library with support for CUDA and HIP. Write portable, high-performance GPU code using a unified interface.
> Write portable, high-performance GPU code in idiomatic C++ — no backend lock-in.

---

## 🚀 Features

- Backend-agnostic GPU runtime abstraction (CUDA, HIP)
- C++17 friendly interface
- Simple device memory management API
- Minimal runtime overhead
- Header-only

---

## 📦 Getting Started

### Prerequisites

- C++17 (or newer) compatible compiler
- CMake 3.20+
- At least one supported GPU backend installed:
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - [ROCm / HIP](https://rocmdocs.amd.com/)

---

### Using GCXX with CPM

You can add GCXX to your project using `CPMAddPackage`:

```cmake
CPMAddPackage(
  NAME gcxx
  GITHUB_REPOSITORY "SriramKatta/GCXX"
  GIT_TAG "DEV"
)
```

You must enable exactly one backend mode when adding `gcxx` (either CUDA or
HIP, but not both).

CUDA example:

```cmake
CPMAddPackage(
  NAME gcxx
  GITHUB_REPOSITORY "SriramKatta/GCXX"
  GIT_TAG "DEV"
  OPTIONS
    "GCXX_CUDA_MODE ON"
)
```

HIP example:

```cmake
CPMAddPackage(
  NAME gcxx
  GITHUB_REPOSITORY "SriramKatta/GCXX"
  GIT_TAG "DEV"
  OPTIONS
    "GCXX_HIP_MODE ON"
)
```

After adding the package, link your executable with:

```cmake
target_link_libraries(exe-main PRIVATE gcxx::gcxx)
```

---

### Building the Examples

GCXX uses CMake workflows and presets for a simple, unified build experience.

#### Build NVIDIA GPU

```bash
cmake --workflow --preset all-cuda-release-examples
```

Executables will be available in:

```bash
build/cudabin-release/
```

#### Build for AMD GPU examples  

```bash
cmake --workflow --preset all-hip-release-examples
```

Executables will be available in:

```bash
build/hipbin-release/
```

---

## 📄 License

GCXX is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE) for the full text.

Each source file carries an [SPDX](https://spdx.dev/) identifier (`// SPDX-License-Identifier: GPL-3.0-or-later`) so the license can be verified automatically with tools like [reuse](https://reuse.software).
