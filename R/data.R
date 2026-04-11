# Data loading utilities
#
# Minimal dataset/dataloader/sampler implementation.
# Adapted from torch R package (MIT, Daniel Falbel).

# ---- Dataset ----

#' Create a dataset
#'
#' @param name Dataset class name.
#' @param ... Named methods: initialize, .getitem, .length.
#' @return A dataset generator function.
#' @export
dataset <- function(name = NULL, ...) {
  methods <- list(...)
  function(...) {
    self <- new.env(parent = emptyenv())
    self$.class <- name %||% "dataset"
    for (nm in names(methods)) self[[nm]] <- methods[[nm]]
    if (!is.null(self$initialize)) {
      environment(self$initialize) <- self
      self$initialize(...)
    }
    if (!is.null(self$.getitem)) environment(self$.getitem) <- self
    if (!is.null(self$.length)) environment(self$.length) <- self
    class(self) <- c(name, "torch_dataset")
    self
  }
}

#' Create a dataset from tensors
#'
#' @param ... Tensors. Each must have the same first dimension.
#' @return A dataset object.
#' @export
tensor_dataset <- function(...) {
  tensors <- list(...)
  n <- tensors[[1]]$size()[1]
  ds <- new.env(parent = emptyenv())
  ds$tensors <- tensors
  ds$.getitem <- function(i) lapply(ds$tensors, function(t) t[i])
  ds$.length <- function() n
  class(ds) <- c("tensor_dataset", "torch_dataset")
  ds
}

#' Subset of a dataset
#'
#' @param dataset A torch_dataset.
#' @param indices Integer vector of indices.
#' @return A dataset object.
#' @export
dataset_subset <- function(dataset, indices) {
  ds <- new.env(parent = emptyenv())
  ds$.parent <- dataset
  ds$.indices <- indices
  ds$.getitem <- function(i) ds$.parent$.getitem(ds$.indices[i])
  ds$.length <- function() length(ds$.indices)
  class(ds) <- c("dataset_subset", "torch_dataset")
  ds
}

#' @export
length.torch_dataset <- function(x) x$.length()

#' @export
`[.torch_dataset` <- function(x, i) x$.getitem(i)

# ---- Sampler ----

#' Create a data sampler
#'
#' @param data_source A dataset (used for length).
#' @return A sampler object.
#' @export
sampler <- function(data_source) {
  n <- if (inherits(data_source, "torch_dataset")) data_source$.length()
       else length(data_source)
  structure(list(n = n), class = "torch_sampler")
}

# ---- Dataloader ----

#' Create a dataloader
#'
#' @param dataset A torch_dataset object.
#' @param batch_size Batch size. Default 1.
#' @param shuffle Whether to shuffle indices each epoch. Default FALSE.
#' @param drop_last Drop the last incomplete batch. Default FALSE.
#' @return A dataloader object.
#' @export
dataloader <- function(dataset, batch_size = 1L, shuffle = FALSE,
                        drop_last = FALSE) {
  dl <- new.env(parent = emptyenv())
  dl$dataset <- dataset
  dl$batch_size <- as.integer(batch_size)
  dl$shuffle <- shuffle
  dl$drop_last <- drop_last
  dl$.length <- function() {
    n <- dl$dataset$.length()
    if (dl$drop_last) n %/% dl$batch_size
    else ceiling(n / dl$batch_size)
  }
  class(dl) <- "dataloader"
  dl
}

#' @export
length.dataloader <- function(x) x$.length()

#' Create an iterator from a dataloader
#'
#' @param dl A dataloader object.
#' @return An iterator environment with .next() method.
#' @export
dataloader_make_iter <- function(dl) {
  n <- dl$dataset$.length()
  indices <- if (dl$shuffle) sample.int(n) else seq_len(n)
  iter <- new.env(parent = emptyenv())
  iter$.pos <- 1L
  iter$.indices <- indices
  iter$.dl <- dl
  iter$.next <- function() {
    if (iter$.pos > length(iter$.indices)) return(NULL)
    end <- min(iter$.pos + iter$.dl$batch_size - 1L, length(iter$.indices))
    if (iter$.dl$drop_last && (end - iter$.pos + 1L) < iter$.dl$batch_size) return(NULL)
    batch_idx <- iter$.indices[iter$.pos:end]
    iter$.pos <- end + 1L
    iter$.dl$dataset$.getitem(batch_idx)
  }
  class(iter) <- "dataloader_iterator"
  iter
}

#' Get next batch from a dataloader iterator
#'
#' @param iter A dataloader_iterator.
#' @return A batch (list of tensors), or NULL if exhausted.
#' @export
dataloader_next <- function(iter) {
  iter$.next()
}

# ---- Iteration helpers ----

#' Iterate with index
#'
#' @param x An iterable (list, vector, or iterator).
#' @return A list of (index, value) pairs.
#' @export
enumerate <- function(x) {
  if (is.list(x)) {
    lapply(seq_along(x), function(i) list(i, x[[i]]))
  } else {
    as.list(x)
  }
}

#' Create an iterator from a dataloader
#'
#' @param x A dataloader.
#' @return An iterator.
#' @export
as_iterator <- function(x) {
  if (inherits(x, "dataloader")) dataloader_make_iter(x)
  else stop("Cannot create iterator from ", class(x)[1])
}

#' Loop over a dataloader (convenience)
#'
#' @param dl A dataloader.
#' @param fn Function to call with each batch.
#' @return Invisible NULL.
#' @export
loop <- function(dl, fn) {
  iter <- dataloader_make_iter(dl)
  repeat {
    batch <- iter$.next()
    if (is.null(batch)) break
    fn(batch)
  }
  invisible(NULL)
}

#' Yield placeholder (for compatibility)
#' @param x Value to yield.
#' @return x
#' @keywords internal
#' @export
yield <- function(x) x

#' Check if object is a dataloader
#' @param x Object to check.
#' @return Logical.
#' @export
is_dataloader <- function(x) inherits(x, "dataloader")

#' Iterable dataset (stub)
#' @param name Dataset name.
#' @param ... Methods.
#' @return A dataset generator.
#' @export
iterable_dataset <- dataset
