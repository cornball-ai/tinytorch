## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local Ubuntu 24.04, R 4.5.0
* win-builder (devel and release)

## Notes

* `SystemRequirements: libtorch` — the package links against the C++ distribution of 'PyTorch'. On Linux the `configure` script locates 'libtorch' via `LIBTORCH_HOME`, a user cache at `~/.local/lib/libtorch`, or the 'torch' R package. When 'libtorch' is not found (including non-Linux Unix platforms), the package installs with a stub backend and `is_available()` returns `FALSE`, so all tests and examples skip gracefully.
* Hard dependencies: only 'Rcpp'.

## Downstream dependencies

None (new package).
