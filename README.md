
# **⚠️ Important Note**  
> Please refer to the **`DEV` branch** for the latest updates.  
> The API is still **under active development** and may change without notice.

---


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
- A supported GPU backend must be installed:
  - **NVIDIA GPUs** require the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - **AMD GPUs** require [ROCm with HIP](https://rocmdocs.amd.com/)

---

### Building the Examples

gpuCXX uses CMake workflows and presets for a simple, unified build experience.

#### Build NVIDIA GPU

```bash
cmake --workflow --preset all-cuda-release-examples
```

Executables will be available in:
```
build/cudabin-release/
```

#### Build for AMD GPU examples 

```bash
cmake --workflow --preset all-hip-release-examples
```

Executables will be available in:
```
build/hipbin-release/
```


