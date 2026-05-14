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

#### Using Development Containers (Recommended)

GCXX provides pre-configured development containers with all necessary tools. This is the recommended approach as it ensures a consistent environment.

**With VS Code:**
1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
2. Open the GCXX repository in VS Code
3. Click "Reopen in Container" when prompted

**From the command line:**
```bash
docker-compose up -d
docker-compose exec dev bash
```

#### Local Development (Without Containers)

If you prefer to set up your environment locally, ensure you have all prerequisites installed (C++17 compiler, CMake, GPU backend).

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

#### Configure the Project

Choose your desired build preset based on your GPU backend:

**For NVIDIA CUDA:**
```bash
cmake --preset gcxx-nvhpc26.3-cuda13.1-gcc13
```

**For AMD ROCm/HIP:**
```bash
cmake --preset gcxx-rocm7.2.3-gcc13
```

Use `cmake --list-presets` to see all available presets.

#### Build Examples

**CUDA Release Build:**
```bash
cmake --workflow --preset all-cuda-release-examples
```

Executables will be available in:
```bash
build/cudabin-release/
```

**HIP Release Build:**
```bash
cmake --workflow --preset all-hip-release-examples
```

Executables will be available in:
```bash
build/hipbin-release/
```

#### Build with Specific Configuration

You can also build specific configurations:

```bash
cmake --build --preset cuda-debug
cmake --build --preset hip-debug
```

### Running Tests

Run the test suite after building:

```bash
ctest --preset cuda-debug  # For CUDA
ctest --preset hip-debug   # For HIP
```

---

## 📄 License

GCXX is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE) for the full text.

Each source file carries an [SPDX](https://spdx.dev/) identifier (`// SPDX-License-Identifier: GPL-3.0-or-later`) so the license can be verified automatically with tools like [reuse](https://reuse.software).
