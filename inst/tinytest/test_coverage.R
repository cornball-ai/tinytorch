# Test codegen coverage against native_functions.yaml
# Only runs locally (needs network + codegen tools)

if (!at_home()) exit_file("Codegen coverage test only runs locally")

library(tinytorch)
source(file.path(find.package("tinytorch"), "..", "..", "..", "tools", "codegen.R"),
       local = TRUE)

# Get pinned version from install_libtorch default
version <- formals(tinytorch::install_libtorch)$version

# Download native_functions.yaml for the pinned version
yaml_url <- sprintf(
  "https://raw.githubusercontent.com/pytorch/pytorch/v%s/aten/src/ATen/native/native_functions.yaml",
  version
)
yaml_path <- tempfile(fileext = ".yaml")
on.exit(unlink(yaml_path))
download.file(yaml_url, yaml_path, quiet = TRUE)

expect_true(file.exists(yaml_path),
            info = "Downloaded native_functions.yaml")
expect_true(file.size(yaml_path) > 100000,
            info = "YAML file is non-trivial size")

# Run codegen dry-run
res <- codegen(yaml_path = yaml_path, dry_run = TRUE)

expect_equal(length(res$generated), 0L,
             info = sprintf("Codegen found %d new ops not yet wrapped: %s",
                            length(res$generated),
                            paste(head(res$generated, 10), collapse = ", ")))
