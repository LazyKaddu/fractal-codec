# Contributing to FractalCodec

Thank you for taking the time to contribute to FractalCodec.

FractalCodec is a mathematically intensive, hardware-accelerated project operating at the intersection of fractal geometry, numerical optimization, CUDA programming, and low-level GPU memory management. Consistent development practices are therefore important for maintaining code quality, correctness, stability, and performance.

## Areas of Contribution

Contributions are welcome across all areas of the project, including:

* **CUDA Optimization:** Improving memory coalescing, shared-memory tiling, occupancy, kernel execution, and parallel reduction efficiency in Domain-Range matching kernels.
* **Core Mathematics:** Improving the mathematical components of the codec, including Domain block classification, affine transformations, error metrics, and search-space reduction techniques.
* **Encoding and Decoding:** Improving the host-side encoding pipeline, fractal reconstruction process, serialization, and deserialization.
* **Documentation:** Improving technical documentation, mathematical explanations, API documentation, and usage examples.
* **Testing:** Expanding the GoogleTest suite to cover edge cases, numerical stability, affine transformations, serialization, CUDA kernels, and memory management.
* **Build System:** Improving CMake configuration, compiler compatibility, CUDA architecture support, and CI workflows.

---

## Development Workflow

### 1. Branching Strategy

Do not commit directly to the `main` branch.

Create a dedicated branch from `main` using one of the following prefixes:

| Prefix      | Purpose                         | Example                             |
| ----------- | ------------------------------- | ----------------------------------- |
| `feature/`  | New functionality               | `feature/cuda-shared-memory-tiling` |
| `bugfix/`   | Bug fixes                       | `bugfix/cuda-memory-leak`           |
| `docs/`     | Documentation changes           | `docs/math-proof-update`            |
| `test/`     | Test additions or modifications | `test/affine-transform-tests`       |
| `refactor/` | Internal code restructuring     | `refactor/encoder-pipeline`         |
| `perf/`     | Performance improvements        | `perf/domain-search-kernel`         |

Create a branch using:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

Keep branches focused on a single logical change whenever possible.

---

## 2. Code Style and Formatting

FractalCodec targets **C++17** and modern **CUDA** toolchains.

All C++ and CUDA source code must be formatted using `clang-format` before submitting a Pull Request.

### Formatting

Run:

```bash
find src include tests \
    \( -iname "*.h" -o -iname "*.hpp" -o -iname "*.cpp" -o -iname "*.cu" -o -iname "*.cuh" \) \
    -print0 | xargs -0 clang-format -i
```

Review the resulting changes before committing.

### Memory Management

Memory ownership must be explicit and follow RAII principles.

Do not use raw ownership patterns such as:

```cpp
new
delete
```

in application-level code when an appropriate RAII abstraction is available.

For CUDA device memory, do not manually manage allocations with:

```cpp
cudaMalloc(...)
cudaFree(...)
```

outside the designated CUDA memory-management abstractions.

Use the project's `CudaBuffer<T>` RAII wrapper where applicable.

For host-side dynamic resources, prefer standard C++ RAII types such as:

```cpp
std::unique_ptr
std::shared_ptr
std::vector
std::array
```

Manual resource management should only be used when there is a clear technical reason and ownership semantics are documented.

---

## 3. Building the Project

The project must be built using **CMake**.

Do not add Makefiles or Visual Studio solution files as alternatives to the project's CMake-based build system.

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

For Windows environments where `nproc` is unavailable:

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Contributors should verify that the project builds successfully before submitting a Pull Request.

---

## 4. Running the Test Suite

All changes must pass the existing test suite before they are merged.

FractalCodec uses **GoogleTest** for automated testing.

The test suite is designed to distinguish between host-side tests and hardware-dependent CUDA tests.

### NVIDIA GPU Available

When a supported NVIDIA GPU is available, the test suite can execute:

* CUDA kernel tests
* GPU memory allocation tests
* Device-side numerical operations
* CUDA-specific integration tests
* Host-side CPU and mathematical tests

### NVIDIA GPU Unavailable

When an NVIDIA GPU is unavailable, hardware-dependent tests should be skipped safely.

Host-side tests, mathematical tests, serialization tests, and other tests that do not require CUDA hardware should still execute.

Run the test suite with:

```bash
ctest --test-dir build --output-on-failure
```

Before submitting a Pull Request, verify that the test results contain no unexpected failures.

---

## 5. CUDA Development Guidelines

CUDA code is performance-critical and should be developed with both correctness and GPU architecture in mind.

When modifying CUDA kernels, consider:

* Global memory coalescing
* Shared-memory usage
* Register pressure
* Warp divergence
* Occupancy
* Memory bandwidth
* Synchronization overhead
* Kernel launch configuration
* Atomic operations
* Numerical precision
* Device memory lifetime

Avoid introducing synchronization or memory transfers without measuring their impact.

When optimizing an existing kernel, include benchmark results where practical.

For example:

```text
Before:
Frame encoding: 42 ms

After:
Frame encoding: 28 ms

Hardware:
NVIDIA RTX 3060

Input:
1920 × 1080 frame
Range size: 4 × 4
Domain size: 8 × 8
```

Performance claims should be reproducible and should identify the relevant hardware and workload.

---

## 6. Numerical and Mathematical Changes

Changes to the mathematical core of FractalCodec require particular care.

When modifying algorithms involving:

* Affine transformations
* Contrast and brightness estimation
* RMSE calculations
* Domain-Range matching
* Fractal transformations
* Block classification
* Iterative reconstruction

contributors should provide appropriate tests demonstrating correctness.

For mathematically non-trivial changes, include a short explanation of:

1. The mathematical formulation.
2. The motivation for the change.
3. Any assumptions being made.
4. Numerical precision considerations.
5. Expected effects on encoding or reconstruction quality.

Changes that alter codec behavior should include regression tests where possible.

---

## 7. Commit Guidelines

Use the **Conventional Commits** format for commit messages.

Examples:

```text
feat: add CUDA domain search kernel
fix: prevent double-free in decoder
perf: optimize shared memory access
refactor: simplify encoder pipeline
test: add affine transformation tests
docs: update codec architecture documentation
build: update CUDA architecture configuration
```

Keep commits focused and avoid combining unrelated changes into a single commit.

A commit should ideally represent one logical change that can be reviewed independently.

---

# Submitting a Pull Request

When your changes are ready, open a Pull Request against the `main` branch.

## Pull Request Title

Use the Conventional Commits format.

Examples:

```text
feat: add CUDA domain classification
fix: resolve decoder memory ownership issue
perf: optimize range-domain matching
test: add PIFS reconstruction tests
```

## Pull Request Description

Clearly describe:

* What was changed.
* Why the change was necessary.
* How the implementation works.
* Any relevant design decisions.
* Tests that were added or modified.
* Performance implications, if applicable.
* Any known limitations.

For CUDA performance changes, include benchmark results when applicable.

For example:

```text
Reduced average frame encoding time from 42 ms to 28 ms
on an NVIDIA RTX 3060 using a 1920 × 1080 input frame.
```

## Testing

Before submitting the Pull Request, run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Ensure that all expected tests pass.

## Continuous Integration

GitHub Actions will build the project and execute the applicable automated tests.

Review the CI results after opening the Pull Request.

A Pull Request should not be considered ready for review until the CI workflow completes successfully.

---

# Reporting Bugs

If you discover a bug, particularly a crash, memory leak, incorrect reconstruction, CUDA error, or segmentation fault, open an Issue using the appropriate issue template.

Please provide enough information to reproduce the problem.

Include the following where applicable:

### 1. Operating System

Example:

```text
Ubuntu 22.04
```

### 2. GPU

Example:

```text
NVIDIA RTX 3060 12 GB
```

### 3. NVIDIA Driver Version

Example:

```text
Driver: 555.xx
```

### 4. CUDA Version

Provide the output of:

```bash
nvcc --version
```

### 5. Reproduction Steps

Provide the smallest possible command, input, or code example that reproduces the issue.

For example:

```bash
./build/fractal_cli \
    --encode \
    --input sample.mp4 \
    --output output.frac \
    --range-size 4 \
    --domain-size 8
```

If the issue depends on a particular input file, provide a minimal reproducible sample whenever possible.

### 6. Error Output

Include the relevant compiler, runtime, CUDA, or application error messages.

For example:

```text
CUDA error: an illegal memory access was encountered
```

Avoid posting unrelated logs that make the issue difficult to diagnose.

---

# Code Review Expectations

Pull Requests are reviewed for:

* Correctness
* Memory safety
* Numerical stability
* CUDA correctness
* Performance
* Maintainability
* Test coverage
* API consistency
* Documentation quality
* Compatibility with the existing build system

Reviewers may request changes before a Pull Request is merged.

Contributors are expected to address review feedback or provide a technical justification for retaining the existing implementation.

---

# General Guidelines

Please follow these principles when contributing:

1. Keep changes focused and minimal.
2. Prefer clear and maintainable implementations over unnecessary complexity.
3. Avoid premature optimization.
4. Measure performance before and after performance-related changes.
5. Add tests for new functionality and bug fixes.
6. Document non-obvious mathematical or CUDA-specific logic.
7. Do not introduce unmanaged resource ownership without a strong justification.
8. Preserve existing APIs unless a breaking change is explicitly required.
9. Keep platform-specific code isolated where possible.
10. Ensure that changes can be built and tested through the project's CMake workflow.

---

# License

By contributing to FractalCodec, you agree that your contributions will be licensed under the same license as the project.

FractalCodec is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the complete license text.
