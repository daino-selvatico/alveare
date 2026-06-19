# tests/

Correctness and performance tests.

- Correctness: every kernel compared against its CPU/numpy reference (`tools/ref/`) within a documented tolerance. Real weights (from local GGUFs) + random inputs.
- `bench/` — microbenchmarks (NPU vs CPU), with shape and machine recorded alongside each number.

Principle: a kernel is not "done" without a passing correctness test. See `../CONTRIBUTING.md`.

Empty until M1.
