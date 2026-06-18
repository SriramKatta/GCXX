# Development Containers

GCXX provides pre-configured development containers with all necessary tools. This is the recommended approach as it ensures a consistent environment.

## Available Configurations

| Directory | Toolchain |
|---|---|
| `nvhpc25.3-cuda12.8-gcc13/` | NVHPC 25.3, CUDA 12.8, GCC 13 |
| `nvhpc25.7-cuda12.9-gcc13/` | NVHPC 25.7, CUDA 12.9, GCC 13 |
| `nvhpc25.11-cuda13.0-gcc13/` | NVHPC 25.11, CUDA 13.0, GCC 13 |
| `nvhpc26.3-cuda13.1-gcc13/` | NVHPC 26.3, CUDA 13.1, GCC 13 |
| `cuda13.2.1-runtime-gcc13/` | CUDA 13.2.1 runtime, GCC 13 |
| `cuda13.3.0-runtime-gcc13/` | CUDA 13.3.0 runtime, GCC 13 |
| `rocm7.1.1-complete/` | ROCm 7.1.1, HIP |
| `rocm7.2.3-complete/` | ROCm 7.2.3, HIP |

## Usage

### With VS Code

1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
2. Open the GCXX repository in VS Code
3. Click "Reopen in Container" when prompted (select the desired config)

### From the Command Line

```bash
# Start the container (CUDA 12.8 example)
devcontainer up --workspace-folder . \
  --config .devcontainer/nvhpc25.3-cuda12.8-gcc13/devcontainer.json

# Open a bash shell inside the container
docker exec -it nvhpc-25.3cuda12.8-devel-gcc13 bash

# Or use devcontainer exec
devcontainer exec --workspace-folder . \
  --config .devcontainer/nvhpc25.3-cuda12.8-gcc13/devcontainer.json bash

# One-liner: up and shell
devcontainer up --workspace-folder . \
  --config .devcontainer/nvhpc25.3-cuda12.8-gcc13/devcontainer.json; \
docker exec -it nvhpc-25.3cuda12.8-devel-gcc13 bash
```
