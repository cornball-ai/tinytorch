# ---- Minimal nn_module system ----
# Uses environments to mimic R6-style private fields expected by torchlang tracer.
# Key interface: mod$.__enclos_env__$private has parameters_, modules_, buffers_

#' Create an nn_module class
#'
#' @param classname Character name for the module class (optional,
#'   defaults to "nn_module").
#' @param ... Named functions: initialize, forward, and any other methods.
#' @return A constructor function that creates module instances.
#' @export
nn_module <- function(classname = NULL, ...) {
  if (is.null(classname)) classname <- "nn_module"
  methods <- list(...)

  constructor <- function(...) {
    # Private environment holds parameters_, modules_, buffers_
    private <- new.env(parent = emptyenv())
    private$parameters_ <- list()
    private$modules_ <- list()
    private$buffers_ <- list()
    private$.module_name <- classname

    # Public environment (the module itself)
    self <- new.env(parent = emptyenv())

    # Mimic R6 enclosing env structure for tracer compatibility
    enclos <- new.env(parent = emptyenv())
    enclos$private <- private
    self$`.__enclos_env__` <- enclos

    # Helper to register parameters (called during initialize)
    self$register_parameter <- function(name, tensor) {
      private$parameters_[[name]] <- tensor
      self[[name]] <- tensor
    }

    self$register_buffer <- function(name, tensor) {
      private$buffers_[[name]] <- tensor
      self[[name]] <- tensor
    }

    self$register_module <- function(name, module) {
      private$modules_[[name]] <- module
      self[[name]] <- module
    }

    # Training mode (no-op for now, but tracer checks it)
    self$training <- TRUE
    self$eval <- function() { self$training <- FALSE; invisible(NULL) }
    self$train <- function(mode = TRUE) { self$training <- mode; invisible(NULL) }

    # Bind all user methods with self available in environment
    for (nm in names(methods)) {
      fn <- methods[[nm]]
      if (is.function(fn)) {
        # Bind self in the method's environment so it's accessible as a free variable
        e <- new.env(parent = environment(fn))
        e$self <- self
        environment(fn) <- e
        self[[nm]] <- fn
      }
    }

    # Run initialize
    if (!is.null(methods$initialize)) {
      self$initialize(...)
    }

    # Auto-register: scan fields set during initialize for modules/tensors
    skip_names <- c("register_parameter", "register_buffer", "register_module",
                    "training", "eval", "train", ".__enclos_env__",
                    names(methods))
    for (nm in ls(self, all.names = FALSE)) {
      if (nm %in% skip_names) next
      val <- self[[nm]]
      if (inherits(val, "nn_module") && !nm %in% names(private$modules_)) {
        private$modules_[[nm]] <- val
      } else if (inherits(val, "torch_tensor") && !nm %in% names(private$parameters_) &&
                 !nm %in% names(private$buffers_)) {
        private$parameters_[[nm]] <- val
      }
    }

    # Make module callable: m(x) calls m$forward(x)
    # Wrap in a function that dispatches to forward
    callable <- function(...) self$forward(...)
    # Store the module env so $.nn_module can access fields
    attr(callable, ".module_env") <- self
    class(callable) <- c(classname, "nn_module")
    callable
  }

  class(constructor) <- "nn_module_generator"
  constructor
}

#' @export
`$.nn_module` <- function(x, name) {
  env <- attr(x, ".module_env")
  if (!is.null(env) && exists(name, envir = env, inherits = FALSE)) {
    env[[name]]
  } else {
    NULL
  }
}

#' @export
`$<-.nn_module` <- function(x, name, value) {
  env <- attr(x, ".module_env")
  env[[name]] <- value
  x
}


#' Linear layer
#'
#' @param in_features Integer, size of each input sample.
#' @param out_features Integer, size of each output sample.
#' @param bias Logical, whether to include a bias term.
#' @return An nn_module instance.
#' @export
nn_linear <- function(in_features, out_features, bias = TRUE) {
  mod <- nn_module("nn_linear",
    initialize = function(in_features, out_features, bias) {
      w <- torch_randn(c(out_features, in_features))
      self$register_parameter("weight", w)
      if (bias) {
        b <- torch_zeros(c(out_features))
        self$register_parameter("bias", b)
      }
      self$in_features <- in_features
      self$out_features <- out_features
    },
    forward = function(input) {
      torch_linear(input, self$weight, self$bias)
    }
  )
  mod(in_features, out_features, bias)
}


#' ReLU activation module
#'
#' @return An nn_module instance.
#' @export
nn_relu <- function() {
  nn_module(
    "relu",
    forward = function(input) {
      input$relu()
    }
  )()
}


#' GELU activation module
#'
#' @return An nn_module instance.
#' @export
nn_gelu <- function() {
  nn_module(
    "gelu",
    forward = function(input) {
      nnf_gelu(input)
    }
  )()
}


#' Sequential container
#'
#' @param ... nn_module instances to chain sequentially.
#' @return An nn_module instance.
#' @export
nn_sequential <- function(...) {
  layers <- list(...)

  mod <- nn_module("nn_sequential",
    initialize = function(layers) {
      for (i in seq_along(layers)) {
        nm <- as.character(i - 1L)  # 0-based like PyTorch
        self$register_module(nm, layers[[i]])
      }
      self$.layers <- layers
    },
    forward = function(input) {
      x <- input
      for (layer in self$.layers) {
        x <- layer$forward(x)
      }
      x
    }
  )
  mod(layers)
}


#' Layer normalization module
#'
#' @param normalized_shape Integer or integer vector of the shape
#'   to normalize over (typically the last dimension size).
#' @param eps Small constant for numerical stability.
#' @return An nn_module instance.
#' @export
nn_layer_norm <- function(normalized_shape, eps = 1e-5) {
  mod <- nn_module("nn_layer_norm",
    initialize = function(normalized_shape, eps) {
      if (length(normalized_shape) == 1L) {
        normalized_shape <- as.integer(normalized_shape)
      }
      self$normalized_shape <- normalized_shape
      self$eps <- eps
      self$register_parameter("weight", torch_ones(normalized_shape))
      self$register_parameter("bias", torch_zeros(normalized_shape))
    },
    forward = function(input) {
      nnf_layer_norm(input, self$normalized_shape, self$weight, self$bias, self$eps)
    }
  )
  mod(normalized_shape, eps)
}
