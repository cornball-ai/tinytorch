## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.
* NOTE: Compilation used non-portable flags inherited from system R
  (`-mno-omit-leaf-frame-pointer`). These are Ubuntu 24.04 defaults,
  not set by the package.

## Test environments

* local Ubuntu 24.04, R 4.5.3
* GitHub Actions (ubuntu-latest, macos-latest)
* win-builder (R-devel)

## Notes

* `SystemRequirements: libtorch` -- the package links against the C++
  distribution of 'PyTorch'. On Linux the `configure` script locates
  'libtorch' via `LIBTORCH_HOME`, a user cache at
  `~/.local/lib/libtorch`, or the 'torch' R package. When 'libtorch'
  is not found (including non-Linux Unix platforms), the package installs
  with a stub backend and `is_available()` returns `FALSE`, so all tests
  and examples skip gracefully.
* Hard dependencies: only 'Rcpp'.

## Downstream dependencies

None (new package).
