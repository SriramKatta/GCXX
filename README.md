# GCXX

> Refer to the DEV branch for the latest updates. The code is currently in development and comes with no guarantees that the API will remain the same.

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

Choose one of the following development approaches:

**Option 1: Using Development Containers (Recommended)**
- Docker
- Internet connection
- (All dependencies are pre-configured in the container)

**Option 2: Local Development**
- C++17 (or newer) compatible compiler
- CMake 3.20+
- At least one supported GPU backend installed:
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - [ROCm / HIP](https://rocmdocs.amd.com/)

### Development Environment Setup

Pre-configured development containers are available in `.devcontainer/`. See [.devcontainer/README.md](.devcontainer/README.md) for usage instructions and available configurations.

For local development (without containers), ensure you have a C++17 compiler, CMake 3.20+, and at least one supported GPU backend (CUDA or HIP) installed.

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

### Building the Project

GCXX uses CMake with presets for a simple, unified build experience across both dev containers and local development.

#### Configure and Build

GCXX provides presets for different configurations. Use workflow presets to configure and build in one step:

**CUDA Release Examples:**
```bash
cmake --workflow --preset workflow_release_cuda_examples
```

**CUDA Debug Examples:**
```bash
cmake --workflow --preset workflow_debug_cuda_examples
```

**HIP Release Examples:**
```bash
cmake --workflow --preset workflow_release_hip_examples
```

**HIP Debug Examples:**
```bash
cmake --workflow --preset workflow_debug_hip_examples
```

Executables will be available in the appropriate `build/` subdirectory.

#### Alternative: Configure and Build Separately

You can also configure and build in separate steps:

**Configure:**
```bash
cmake --preset release_cuda_examples  # or debug_cuda_examples, release_hip_examples, etc.
```

**Build:**
```bash
cmake --build --preset build_release_cuda_examples
```

**For full builds (examples + tests):**
```bash
cmake --workflow --preset workflow_release_cuda_full
cmake --workflow --preset workflow_debug_hip_full
```

Use `cmake --list-presets` to see all available configure, build, and workflow presets.

### Running Tests

Run tests using test presets:

```bash
ctest --preset test_release_cuda   # CUDA Release tests
ctest --preset test_debug_cuda     # CUDA Debug tests
ctest --preset test_release_hip    # HIP Release tests
ctest --preset test_debug_hip      # HIP Debug tests
```

---

## 📄 License

GCXX is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE) for the full text.

Each source file carries an [SPDX](https://spdx.dev/) identifier (`// SPDX-License-Identifier: GPL-3.0-or-later`) so the license can be verified automatically with tools like [reuse](https://reuse.software).
