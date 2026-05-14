# Contributing to GCXX

Thank you for your interest in contributing to GCXX! We welcome contributions from the community to help make this GPU runtime abstraction library better.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Docker
- Internet connection

All development is done through development containers, which provide a pre-configured environment with all necessary tools and GPU backend support.

### Fork & Clone the Repository

1. Fork the GCXX GitHub Repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GCXX.git
   cd GCXX
   ```

### Set Up Development Environment

GCXX uses Development Containers to provide a consistent development environment for both local development and CI. Contributors are strongly encouraged to use these containers as they simplify environment setup.

**Using VS Code:**
- Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- Open the repository folder in VS Code
- Click the "Reopen in Container" button when prompted, or use the command palette (`Ctrl+Shift+P`) and select "Remote-Containers: Reopen in Container"

**Using the command line:**
```bash
docker-compose up -d
docker-compose exec dev bash
```

The development container includes:
- C++17+ compatible compiler
- CMake 3.20+
- CUDA Toolkit
- ROCm / HIP support

### Building the Project

GCXX uses CMake with presets for configuration. Use workflow presets to configure and build in one step:

**For CUDA Debug:**
```bash
cmake --workflow --preset workflow_debug_cuda_examples
```

**For CUDA Release:**
```bash
cmake --workflow --preset workflow_release_cuda_examples
```

**For HIP Debug:**
```bash
cmake --workflow --preset workflow_debug_hip_examples
```

**For HIP Release:**
```bash
cmake --workflow --preset workflow_release_hip_examples
```

Executables will be available in the `build/` subdirectory.

### Running Tests

Run the test suite to verify your setup:
```bash
ctest --preset test_debug_cuda
# or
ctest --preset test_debug_hip
```

### Running Examples

Build and run examples to verify functionality:
```bash
# Examples are built with the workflow presets above
# Run from the appropriate build directory
./build/*/cudabin-debug/vector_add
./build/*/cudabin-debug/reduction
```

## Development Workflow

### Creating a Branch

Always create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/issue-number-description
```

### Coding Standards

- **C++ Standard**: Use C++17 features; avoid newer standards unless necessary
- **Code Style**: Follow the existing code style in the repository
- **Headers**: Include appropriate headers and keep the library header-only (minimize runtime overhead by accepting longer compile times for optimal generated binaries)
- **Documentation**: Add comments for public APIs and complex logic
- **Tests**: Add tests for new features or bug fixes
- **GPU Code**: Ensure CUDA and HIP implementations are equivalent where applicable

### Testing Your Changes

1. Build with your changes:
   ```bash
   cmake --workflow --preset workflow_debug_cuda_full
   ```

2. Run tests:
   ```bash
   ctest --preset test_debug_cuda -V
   ```

3. Test with both backends if possible:
   ```bash
   cmake --workflow --preset workflow_debug_hip_full
   ctest --preset test_debug_hip -V
   ```

4. Build and run examples to verify functionality:
   ```bash
   # Build examples were completed in step 1
   # Run from the build directory
   ./build/*/cudabin-debug/vector_add
   ```

## Contribution Areas

### Areas We Welcome Contributions To

- **Bug Fixes**: Report and fix issues with existing functionality
- **Examples**: Add new examples demonstrating GCXX features
- **Tests**: Improve test coverage and add unit tests
- **Documentation**: Improve README, API documentation, and comments
- **Performance**: Optimization suggestions and implementations
- **Backend Support**: Extensions for newer(or older) CUDA Toolkit and ROCm versions

### Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: clear description of what was added"
   git commit -m "Fix #123: clear description of what was fixed"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** (PR):
   - Go to the main repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear title and description
   - Reference any related issues

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass in both CUDA and HIP configurations
- Follow the existing code style
- Include a clear description of the changes

## Reporting Issues

If you encounter a bug or have a feature request:

1. Check if the issue already exists on GitHub
2. Provide clear steps to reproduce for bugs
3. Include your environment details (GPU, CUDA/HIP version, compiler, OS)
4. Add relevant code snippets or minimal reproducible examples
5. Include error messages and logs if applicable

## License

By contributing to GCXX, you agree that your contributions will be licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Questions?

- Check the [README.md](README.md) for project overview
- Review existing examples in `examples/`
- Look at the test cases in `testing/` for usage patterns
- Open a discussion or issue on GitHub

Thank you for contributing to GCXX! 🚀
