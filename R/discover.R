#' Module Discovery
#'
#' Walks an nn_module's sub-module tree and reports structure.

#' Walk an nn_module's Sub-module Tree
#'
#' Recursively discovers all sub-modules in an nn_module instance.
#' Returns a flat list of module info: path, class, parameter count,
#' forward() argument names, and the live module reference.
#'
#' @param module An nn_module instance
#' @param max_depth Maximum recursion depth (default 10)
#' @return A list of module descriptors, each with:
#'   \item{path}{Dot-separated path from root (e.g., "layers.1.mlp")}
#'   \item{class}{Module class name}
#'   \item{n_params}{Number of parameters}
#'   \item{forward_args}{Formal argument names of forward()}
#'   \item{module}{Live reference to the nn_module}
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' m <- nn_sequential(nn_linear(10, 5), nn_relu())
#' discover_modules(m)
#' }
#' }
#' @export
discover_modules <- function(module, max_depth = 10L) {
  if (!inherits(module, "nn_module")) {
    stop("module must be an nn_module instance", call. = FALSE)
  }

  result <- list()

  walk <- function(mod, path, depth) {
    if (depth > max_depth) return()

    # Get module class name
    cls <- tryCatch({
      # nn_module stores class name as first element
      cl <- class(mod)
      # R torch modules have class like c("nn_module", "nn_module_generator")
      # The actual name is stored in the module
      name <- mod$.__enclos_env__$private$.module_name
      if (is.null(name) || !nzchar(name)) cl[1] else name
    }, error = function(e) class(mod)[1])

    # Get parameter count
    priv <- tryCatch(mod$.__enclos_env__$private, error = function(e) NULL)
    params <- if (!is.null(priv)) priv$parameters_ else list()
    if (is.null(params)) params <- list()

    # Get forward() formals
    fwd_args <- tryCatch({
      fwd <- mod$forward
      if (is.function(fwd)) {
        nms <- names(formals(fwd))
        # Drop 'self' if present (R6 binds it)
        nms[nms != "self"]
      } else character(0)
    }, error = function(e) character(0))

    # Record this module
    info <- list(
      path = path,
      class = cls,
      n_params = length(params),
      param_names = names(params),
      forward_args = fwd_args,
      module = mod
    )
    result[[length(result) + 1L]] <<- info

    # Recurse into sub-modules
    sub_modules <- if (!is.null(priv)) priv$modules_ else list()
    if (is.null(sub_modules)) sub_modules <- list()

    for (name in names(sub_modules)) {
      sub <- sub_modules[[name]]
      if (inherits(sub, "nn_module")) {
        child_path <- if (nzchar(path)) paste0(path, ".", name) else name
        walk(sub, child_path, depth + 1L)
      }
    }

    # Also check nn_module_list members (stored as numbered elements)
    if (inherits(mod, "nn_module")) {
      tryCatch({
        # nn_module_list stores items in private$modules_ with numeric names
        # Already handled above. But some modules store lists differently.
        NULL
      }, error = function(e) NULL)
    }
  }

  walk(module, "", 0L)
  structure(result, class = "module_tree")
}


#' Print Module Tree
#'
#' @param x A module_tree from discover_modules()
#' @param ... Ignored
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' m <- nn_sequential(nn_linear(10, 5), nn_relu())
#' tree <- discover_modules(m)
#' print(tree)
#' }
#' }
#' @return Invisible `x`.
#' @export
print.module_tree <- function(x, ...) {
  cat(sprintf("Module tree: %d modules\n\n", length(x)))
  cat(sprintf("%-40s %-25s %6s  %s\n", "Path", "Class", "Params", "Forward args"))
  cat(paste(rep("-", 90), collapse = ""), "\n")

  for (info in x) {
    path <- if (nzchar(info$path)) info$path else "(root)"
    args <- paste(info$forward_args, collapse = ", ")
    cat(sprintf("%-40s %-25s %6d  (%s)\n",
                substr(path, 1, 40), substr(info$class, 1, 25),
                info$n_params, args))
  }
  invisible(x)
}


#' Find Installed Packages That Depend on torch
#'
#' Searches installed R packages for those that import or depend on torch.
#'
#' @return Character vector of package names
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' find_torch_packages()
#' }
#' }
#' @export
find_torch_packages <- function() {
  db <- utils::installed.packages()
  pkgs <- character(0)
  for (pkg in c("torch", "tinytorch")) {
    deps <- tools::package_dependencies(pkg, db = db,
              reverse = TRUE, which = c("Depends", "Imports", "Suggests"))
    found <- deps[[pkg]]
    if (!is.null(found)) pkgs <- c(pkgs, found)
  }
  # Always include tinytorch itself
  pkgs <- unique(c(pkgs, "tinytorch"))
  sort(pkgs)
}


#' Find nn_module Definitions in an R Package
#'
#' Searches a package's R source files for nn_module definitions.
#' Works on both installed packages and local source directories.
#'
#' @param pkg Package name (string) or path to package source directory
#' @return A data.frame with columns: name, file, exported
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' find_modules_in_package("tinytorch")
#' }
#' }
#' @export
find_modules_in_package <- function(pkg) {
  # Determine source directory
  if (dir.exists(pkg)) {
    r_dir <- file.path(pkg, "R")
    ns_file <- file.path(pkg, "NAMESPACE")
    pkg_name <- basename(pkg)
  } else {
    pkg_path <- find.package(pkg, quiet = TRUE)
    if (length(pkg_path) == 0) {
      stop(sprintf("Package '%s' not found", pkg), call. = FALSE)
    }
    r_dir <- file.path(pkg_path, "R")
    ns_file <- file.path(pkg_path, "NAMESPACE")
    pkg_name <- pkg
  }

  if (!dir.exists(r_dir)) {
    # Installed package — R/ contains .rdb/.rdx, not .R files
    # Try runtime discovery instead
    return(.find_modules_runtime(pkg_name))
  }

  # Find all .R files
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)

  # Search for nn_module definitions
  # Pattern: name <- nn_module(
  pattern <- "^\\s*([a-zA-Z_.][a-zA-Z0-9_.]*)(\\s*<-\\s*)(tinytorch::)?nn_module\\("

  results <- list()
  for (f in r_files) {
    lines <- readLines(f, warn = FALSE)
    matches <- grep(pattern, lines)
    for (m in matches) {
      name <- sub(pattern, "\\1", lines[m])
      name <- trimws(name)
      results[[length(results) + 1L]] <- data.frame(
        name = name,
        file = basename(f),
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(results) == 0) {
    return(data.frame(name = character(0), file = character(0),
                      exported = logical(0), stringsAsFactors = FALSE))
  }

  df <- do.call(rbind, results)

  # Check exports
  exported_names <- character(0)
  if (file.exists(ns_file)) {
    ns_lines <- readLines(ns_file, warn = FALSE)
    export_lines <- grep("^export\\(", ns_lines, value = TRUE)
    exported_names <- sub("^export\\((.*)\\)$", "\\1", export_lines)
  }
  df$exported <- df$name %in% exported_names

  df
}


#' Find nn_modules at Runtime in an Installed Package
#' @param pkg_name Package name string.
#' @keywords internal
.find_modules_runtime <- function(pkg_name) {
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    stop(sprintf("Package '%s' is not installed", pkg_name), call. = FALSE)
  }

  ns <- asNamespace(pkg_name)
  all_names <- ls(ns, all.names = TRUE)

  # Check exports
  exports <- getNamespaceExports(pkg_name)

  results <- list()
  for (nm in all_names) {
    obj <- tryCatch(get(nm, envir = ns, inherits = FALSE), error = function(e) NULL)
    if (!is.null(obj) && inherits(obj, "nn_module")) {
      results[[length(results) + 1L]] <- data.frame(
        name = nm,
        file = NA_character_,
        exported = nm %in% exports,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(results) == 0) {
    return(data.frame(name = character(0), file = character(0),
                      exported = logical(0), stringsAsFactors = FALSE))
  }

  do.call(rbind, results)
}
