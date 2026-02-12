# ---- Minimal nn_module system ----
# Uses environments to mimic R6-style private fields expected by torchlang tracer.
# Key interface: mod$.__enclos_env__$private has parameters_, modules_, buffers_

#' Create an nn_module class
#'
#' @param classname Character name for the module class (optional,
#'   defaults to "nn_module").
#' @param ... Named functions: initialize, forward, and any other methods.
#' @return A constructor function that creates module instances.
#' @examples
#' \donttest{
#' Linear <- nn_module("Linear",
#'   initialize = function(in_f, out_f) {
#'     self$register_parameter("weight", torch_randn(c(out_f, in_f)))
#'   },
#'   forward = function(x) torch_matmul(x, self$weight$t())
#' )
#' m <- Linear(3, 2)
#' m(torch_randn(c(1, 3)))
#' }
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
    self$eval <- function() { self$training <- FALSE; invisible(self) }
    self$train <- function(mode = TRUE) { self$training <- mode; invisible(self) }

    # Device/dtype transfer (recursive over parameters, buffers, sub-modules)
    self$to <- function(device = NULL, dtype = NULL) {
      # Transfer own parameters
      for (nm in names(private$parameters_)) {
        p <- private$parameters_[[nm]]
        if (inherits(p, "torch_tensor")) {
          p_new <- p
          if (!is.null(dtype) && !is.null(device)) {
            p_new <- C_tensor_to_dtype_device(p, unclass(dtype), as.character(device))
          } else if (!is.null(device)) {
            p_new <- C_tensor_to_device(p, as.character(device))
          } else if (!is.null(dtype)) {
            p_new <- C_torch_to_dtype(p, unclass(dtype))
          }
          cls <- class(p)
          class(p_new) <- cls
          private$parameters_[[nm]] <- p_new
          self[[nm]] <- p_new
        }
      }
      # Transfer own buffers
      for (nm in names(private$buffers_)) {
        b <- private$buffers_[[nm]]
        if (inherits(b, "torch_tensor")) {
          b_new <- b
          if (!is.null(dtype) && !is.null(device)) {
            b_new <- C_tensor_to_dtype_device(b, unclass(dtype), as.character(device))
          } else if (!is.null(device)) {
            b_new <- C_tensor_to_device(b, as.character(device))
          } else if (!is.null(dtype)) {
            b_new <- C_torch_to_dtype(b, unclass(dtype))
          }
          cls <- class(b)
          class(b_new) <- cls
          private$buffers_[[nm]] <- b_new
          self[[nm]] <- b_new
        }
      }
      # Recurse into sub-modules
      for (nm in names(private$modules_)) {
        sub <- private$modules_[[nm]]
        # Handle both callable (with .module_env attr) and raw environment modules
        sub_env <- attr(sub, ".module_env")
        if (is.null(sub_env) && is.environment(sub) && !is.null(sub$to)) {
          sub_env <- sub
        }
        if (!is.null(sub_env) && !is.null(sub_env$to)) {
          sub_env$to(device = device, dtype = dtype)
        }
      }
      # Rebuild the parameters snapshot so $parameters reflects the new device/dtype
      self$parameters <- collect_params(self)
      invisible(self)
    }

    # Parameters accessor (returns named list of registered parameters)
    # Note: this is a snapshot; re-access for live view
    self$parameters <- NULL  # Placeholder, updated after initialize

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
                    "training", "eval", "train", "to", "parameters",
                    ".__enclos_env__", names(methods))
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

    # Build recursive parameters list (own + sub-modules)
    collect_params <- function(mod_env) {
      priv <- mod_env$.__enclos_env__$private
      params <- priv$parameters_
      for (nm in names(priv$modules_)) {
        sub_mod <- priv$modules_[[nm]]
        # Handle both callable (with .module_env attr) and raw environment modules
        sub_env <- attr(sub_mod, ".module_env")
        if (is.null(sub_env) && is.environment(sub_mod) &&
            !is.null(sub_mod$`.__enclos_env__`$private)) {
          sub_env <- sub_mod
        }
        if (!is.null(sub_env)) {
          sub_params <- collect_params(sub_env)
          for (pnm in names(sub_params)) {
            params[[paste0(nm, ".", pnm)]] <- sub_params[[pnm]]
          }
        }
      }
      params
    }
    self$parameters <- collect_params(self)

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
  # Auto-register sub-modules and parameters assigned after init
  priv <- env$`.__enclos_env__`$private
  # Check for nn_module callable or raw module environment (from $to() return)
  is_mod <- inherits(value, "nn_module") ||
    (is.environment(value) && !is.null(value$`.__enclos_env__`$private$modules_))
  if (is_mod) {
    priv$modules_[[name]] <- value
  } else if (inherits(value, "nn_parameter")) {
    priv$parameters_[[name]] <- value
  } else if (inherits(value, "nn_buffer")) {
    priv$buffers_[[name]] <- value
  } else if (is.null(value)) {
    # Allow setting to NULL (remove registration)
    priv$modules_[[name]] <- NULL
    priv$parameters_[[name]] <- NULL
    priv$buffers_[[name]] <- NULL
  }
  x
}

#' @export
`[[.nn_module` <- function(x, i) {
  env <- attr(x, ".module_env")
  if (is.character(i)) {
    env[[i]]
  } else if (is.numeric(i)) {
    # Integer index: access sub-modules list (1-indexed)
    mods <- env$`.__enclos_env__`$private$modules_
    if (i >= 1 && i <= length(mods)) {
      mods[[i]]
    } else {
      # Try .modules field (nn_module_list stores here)
      if (exists(".modules", envir = env, inherits = FALSE)) {
        env$.modules[[i]]
      } else {
        stop(sprintf("index %d out of range", i))
      }
    }
  } else {
    stop("invalid subscript type")
  }
}

#' @export
length.nn_module <- function(x) {
  env <- attr(x, ".module_env")
  if (exists(".modules", envir = env, inherits = FALSE)) {
    length(env$.modules)
  } else {
    length(env$`.__enclos_env__`$private$modules_)
  }
}


#' Linear layer
#'
#' @param in_features Integer, size of each input sample.
#' @param out_features Integer, size of each output sample.
#' @param bias Logical, whether to include a bias term.
#' @return An nn_module instance.
#' @examples
#' \donttest{
#' m <- nn_linear(3, 2)
#' m(torch_randn(c(1, 3)))
#' }
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
#' @examples
#' \donttest{
#' m <- nn_relu()
#' m(torch_randn(c(2, 3)))
#' }
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
#' @examples
#' \donttest{
#' m <- nn_gelu()
#' m(torch_randn(c(2, 3)))
#' }
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
#' @examples
#' \donttest{
#' m <- nn_sequential(nn_linear(3, 4), nn_relu(), nn_linear(4, 2))
#' m(torch_randn(c(1, 3)))
#' }
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
#' @examples
#' \donttest{
#' m <- nn_layer_norm(4)
#' m(torch_randn(c(2, 4)))
#' }
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


# ---- nn_parameter / nn_buffer ----

#' Tag a tensor as an nn parameter
#' @param data A torch_tensor.
#' @param requires_grad Ignored (no autograd).
#' @return The tensor with additional class tag.
#' @export
nn_parameter <- function(data, requires_grad = TRUE) {
  class(data) <- c("nn_parameter", class(data))
  data
}

#' Tag a tensor as a buffer (not a parameter)
#' @param data A torch_tensor.
#' @param persistent Ignored.
#' @return The tensor with additional class tag.
#' @export
nn_buffer <- function(data, persistent = TRUE) {
  class(data) <- c("nn_buffer", class(data))
  data
}

# ---- nn_module_list ----

#' List container for nn_modules
#' @param modules A list of nn_module instances.
#' @return An nn_module containing the sub-modules.
#' @export
nn_module_list <- function(modules = list()) {
  mod <- nn_module("nn_module_list",
    initialize = function(modules) {
      for (i in seq_along(modules)) {
        nm <- as.character(i - 1L)
        self$register_module(nm, modules[[i]])
      }
      self$.modules <- modules
    },
    forward = function(...) {
      stop("nn_module_list has no forward method")
    },
    append = function(module) {
      nm <- as.character(length(self$.modules))
      self$register_module(nm, module)
      self$.modules <- c(self$.modules, list(module))
      invisible(NULL)
    }
  )
  m <- mod(modules)
  # Make it iterable like a list
  env <- attr(m, ".module_env")
  env$length <- function() length(env$.modules)
  m
}


# ---- nn_identity ----

#' Identity module (pass-through)
#' @return An nn_module instance.
#' @export
nn_identity <- function() {
  nn_module("nn_identity",
    forward = function(input) input
  )()
}


# ---- nn_dropout ----

#' Dropout module (no-op in eval mode, Rtorch has no autograd)
#' @param p Probability of dropping. Ignored.
#' @param inplace Ignored.
#' @return An nn_module instance.
#' @export
nn_dropout <- function(p = 0.5, inplace = FALSE) {
  nn_module("nn_dropout",
    initialize = function(p) {
      self$p <- p
    },
    forward = function(input) input
  )(p)
}


# ---- Activation modules ----

#' Sigmoid activation module
#' @return An nn_module instance.
#' @export
nn_sigmoid <- function() {
  nn_module("nn_sigmoid",
    forward = function(input) C_torch_sigmoid(input)
  )()
}

#' SiLU (Swish) activation module
#' @return An nn_module instance.
#' @export
nn_silu <- function() {
  nn_module("nn_silu",
    forward = function(input) nnf_silu(input)
  )()
}

#' Tanh activation module
#' @return An nn_module instance.
#' @export
nn_tanh <- function() {
  nn_module("nn_tanh",
    forward = function(input) C_torch_tanh(input)
  )()
}

#' ELU activation module
#' @param alpha Scale for the negative factor. Default 1.0.
#' @return An nn_module instance.
#' @export
nn_elu <- function(alpha = 1.0) {
  nn_module("nn_elu",
    initialize = function(alpha) {
      self$alpha <- alpha
    },
    forward = function(input) nnf_elu(input, self$alpha)
  )(alpha)
}


# ---- nn_embedding ----

#' Embedding lookup module
#' @param num_embeddings Integer, size of the dictionary.
#' @param embedding_dim Integer, size of each embedding vector.
#' @param padding_idx Optional integer. If given, pads output at this index.
#' @return An nn_module instance.
#' @export
nn_embedding <- function(num_embeddings, embedding_dim, padding_idx = NULL) {
  mod <- nn_module("nn_embedding",
    initialize = function(num_embeddings, embedding_dim, padding_idx) {
      self$num_embeddings <- num_embeddings
      self$embedding_dim <- embedding_dim
      self$padding_idx <- padding_idx
      w <- torch_randn(c(num_embeddings, embedding_dim))
      self$register_parameter("weight", w)
    },
    forward = function(input) {
      torch_embedding(self$weight, input)
    }
  )
  mod(num_embeddings, embedding_dim, padding_idx)
}


# ---- nn_conv1d ----

#' 1D convolution module
#' @param in_channels Integer.
#' @param out_channels Integer.
#' @param kernel_size Integer.
#' @param stride Integer. Default 1.
#' @param padding Integer. Default 0.
#' @param dilation Integer. Default 1.
#' @param groups Integer. Default 1.
#' @param bias Logical. Default TRUE.
#' @return An nn_module instance.
#' @export
nn_conv1d <- function(in_channels, out_channels, kernel_size,
                      stride = 1L, padding = 0L, dilation = 1L,
                      groups = 1L, bias = TRUE) {
  mod <- nn_module("nn_conv1d",
    initialize = function(in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, bias) {
      self$in_channels <- in_channels
      self$out_channels <- out_channels
      self$kernel_size <- kernel_size
      self$stride <- stride
      self$padding <- padding
      self$dilation <- dilation
      self$groups <- groups
      w <- torch_randn(c(out_channels, as.integer(in_channels / groups), kernel_size))
      self$register_parameter("weight", w)
      if (bias) {
        b <- torch_zeros(c(out_channels))
        self$register_parameter("bias", b)
      }
    },
    forward = function(input) {
      torch_conv1d(input, self$weight, self$bias,
                   self$stride, self$padding,
                   self$dilation, self$groups)
    }
  )
  mod(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
}


# ---- nn_conv_transpose1d ----

#' 1D transposed convolution module
#' @param in_channels Integer.
#' @param out_channels Integer.
#' @param kernel_size Integer.
#' @param stride Integer. Default 1.
#' @param padding Integer. Default 0.
#' @param output_padding Integer. Default 0.
#' @param groups Integer. Default 1.
#' @param bias Logical. Default TRUE.
#' @param dilation Integer. Default 1.
#' @return An nn_module instance.
#' @export
nn_conv_transpose1d <- function(in_channels, out_channels, kernel_size,
                                stride = 1L, padding = 0L, output_padding = 0L,
                                groups = 1L, bias = TRUE, dilation = 1L) {
  mod <- nn_module("nn_conv_transpose1d",
    initialize = function(in_channels, out_channels, kernel_size,
                          stride, padding, output_padding, groups, bias, dilation) {
      self$in_channels <- in_channels
      self$out_channels <- out_channels
      self$kernel_size <- kernel_size
      self$stride <- stride
      self$padding <- padding
      self$output_padding <- output_padding
      self$groups <- groups
      self$dilation <- dilation
      w <- torch_randn(c(in_channels, as.integer(out_channels / groups), kernel_size))
      self$register_parameter("weight", w)
      if (bias) {
        b <- torch_zeros(c(out_channels))
        self$register_parameter("bias", b)
      }
    },
    forward = function(input) {
      C_torch_conv_transpose1d(input, self$weight, self$bias,
            as.integer(self$stride), as.integer(self$padding),
            as.integer(self$output_padding), as.integer(self$groups),
            as.integer(self$dilation))
    }
  )
  mod(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
}


# ---- nn_conv2d ----

#' 2D convolution module
#' @param in_channels Integer.
#' @param out_channels Integer.
#' @param kernel_size Integer or vector of 2.
#' @param stride Integer or vector of 2. Default 1.
#' @param padding Integer or vector of 2. Default 0.
#' @param dilation Integer or vector of 2. Default 1.
#' @param groups Integer. Default 1.
#' @param bias Logical. Default TRUE.
#' @return An nn_module instance.
#' @export
nn_conv2d <- function(in_channels, out_channels, kernel_size,
                      stride = 1L, padding = 0L, dilation = 1L,
                      groups = 1L, bias = TRUE) {
  mod <- nn_module("nn_conv2d",
    initialize = function(in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, bias) {
      self$in_channels <- in_channels
      self$out_channels <- out_channels
      ks <- if (length(kernel_size) == 1L) rep(as.integer(kernel_size), 2L) else as.integer(kernel_size)
      self$kernel_size <- ks
      self$stride <- if (length(stride) == 1L) rep(as.integer(stride), 2L) else as.integer(stride)
      self$padding <- if (length(padding) == 1L) rep(as.integer(padding), 2L) else as.integer(padding)
      self$dilation <- if (length(dilation) == 1L) rep(as.integer(dilation), 2L) else as.integer(dilation)
      self$groups <- groups
      w <- torch_randn(c(out_channels, as.integer(in_channels / groups), ks[1], ks[2]))
      self$register_parameter("weight", w)
      if (bias) {
        b <- torch_zeros(c(out_channels))
        self$register_parameter("bias", b)
      }
    },
    forward = function(input) {
      C_torch_conv2d(input, self$weight, self$bias,
            self$stride, self$padding, self$dilation,
            as.integer(self$groups))
    }
  )
  mod(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
}


# ---- nn_batch_norm1d / nn_batch_norm2d ----

#' 1D batch normalization module
#' @param num_features Integer.
#' @param eps Double. Default 1e-5.
#' @param momentum Double. Default 0.1.
#' @param affine Logical. Default TRUE.
#' @param track_running_stats Logical. Default TRUE.
#' @return An nn_module instance.
#' @export
nn_batch_norm1d <- function(num_features, eps = 1e-5, momentum = 0.1,
                            affine = TRUE, track_running_stats = TRUE) {
  mod <- nn_module("nn_batch_norm1d",
    initialize = function(num_features, eps, momentum, affine, track_running_stats) {
      self$num_features <- num_features
      self$eps <- eps
      self$momentum <- momentum
      self$affine <- affine
      self$track_running_stats <- track_running_stats
      if (affine) {
        self$register_parameter("weight", torch_ones(c(num_features)))
        self$register_parameter("bias", torch_zeros(c(num_features)))
      }
      if (track_running_stats) {
        self$register_buffer("running_mean", torch_zeros(c(num_features)))
        self$register_buffer("running_var", torch_ones(c(num_features)))
        self$register_buffer("num_batches_tracked",
                             torch_tensor(0L, dtype = torch_long))
      }
    },
    forward = function(input) {
      C_torch_batch_norm(input,
            self$weight, self$bias,
            self$running_mean, self$running_var,
            FALSE,  # training=FALSE for inference
            self$momentum, self$eps, FALSE)
    }
  )
  mod(num_features, eps, momentum, affine, track_running_stats)
}

#' 2D batch normalization module
#' @param num_features Integer.
#' @param eps Double. Default 1e-5.
#' @param momentum Double. Default 0.1.
#' @param affine Logical. Default TRUE.
#' @param track_running_stats Logical. Default TRUE.
#' @return An nn_module instance.
#' @export
nn_batch_norm2d <- function(num_features, eps = 1e-5, momentum = 0.1,
                            affine = TRUE, track_running_stats = TRUE) {
  # Same as batch_norm1d — at::batch_norm handles both
  nn_batch_norm1d(num_features, eps, momentum, affine, track_running_stats)
}


# ---- nn_lstm ----

#' LSTM module
#' @param input_size Integer.
#' @param hidden_size Integer.
#' @param num_layers Integer. Default 1.
#' @param batch_first Logical. Default TRUE.
#' @param dropout Double. Default 0.
#' @param bidirectional Logical. Default FALSE.
#' @param bias Logical. Default TRUE.
#' @return An nn_module instance.
#' @export
nn_lstm <- function(input_size, hidden_size, num_layers = 1L,
                    batch_first = TRUE, dropout = 0, bidirectional = FALSE,
                    bias = TRUE) {
  mod <- nn_module("nn_lstm",
    initialize = function(input_size, hidden_size, num_layers,
                          batch_first, dropout, bidirectional, bias) {
      self$input_size <- input_size
      self$hidden_size <- hidden_size
      self$num_layers <- num_layers
      self$batch_first <- batch_first
      self$dropout <- dropout
      self$bidirectional <- bidirectional
      self$bias <- bias

      num_directions <- if (bidirectional) 2L else 1L
      for (layer in seq_len(num_layers)) {
        for (dir in seq_len(num_directions)) {
          suffix <- if (layer == 1L && dir == 1L) "" else
                    if (dir == 2L && layer == 1L) "_reverse" else
                    paste0("_l", layer - 1L, if (dir == 2L) "_reverse" else "")
          in_sz <- if (layer == 1L) input_size else hidden_size * num_directions
          nm_ih <- paste0("weight_ih_l", layer - 1L,
                          if (dir == 2L) "_reverse" else "")
          nm_hh <- paste0("weight_hh_l", layer - 1L,
                          if (dir == 2L) "_reverse" else "")
          self$register_parameter(nm_ih, torch_randn(c(4L * hidden_size, in_sz)))
          self$register_parameter(nm_hh, torch_randn(c(4L * hidden_size, hidden_size)))
          if (bias) {
            nm_bih <- paste0("bias_ih_l", layer - 1L,
                             if (dir == 2L) "_reverse" else "")
            nm_bhh <- paste0("bias_hh_l", layer - 1L,
                             if (dir == 2L) "_reverse" else "")
            self$register_parameter(nm_bih, torch_zeros(c(4L * hidden_size)))
            self$register_parameter(nm_bhh, torch_zeros(c(4L * hidden_size)))
          }
        }
      }
    },
    forward = function(input, hx = NULL) {
      # Collect all weight tensors in order expected by at::lstm
      num_directions <- if (self$bidirectional) 2L else 1L
      params <- list()
      for (layer in seq_len(self$num_layers)) {
        for (dir in seq_len(num_directions)) {
          suffix <- paste0("_l", layer - 1L,
                           if (dir == 2L) "_reverse" else "")
          params <- c(params, list(
            self[[paste0("weight_ih", suffix)]],
            self[[paste0("weight_hh", suffix)]],
            if (self$bias) self[[paste0("bias_ih", suffix)]],
            if (self$bias) self[[paste0("bias_hh", suffix)]]
          ))
        }
      }
      C_torch_lstm(input, hx, params,
            as.logical(self$bias), as.integer(self$num_layers),
            as.double(self$dropout), as.logical(self$batch_first),
            as.logical(self$bidirectional))
    }
  )
  mod(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, bias)
}


# ---- nn_upsample ----

#' Upsample module
#' @param size Target size (integer or integer vector).
#' @param scale_factor Scale factor (double or double vector).
#' @param mode Upsampling mode. Default "nearest".
#' @return An nn_module instance.
#' @export
nn_upsample <- function(size = NULL, scale_factor = NULL, mode = "nearest") {
  mod <- nn_module("nn_upsample",
    initialize = function(size, scale_factor, mode) {
      self$size <- size
      self$scale_factor <- scale_factor
      self$mode <- mode
    },
    forward = function(input) {
      nnf_interpolate(input, size = self$size,
                      scale_factor = self$scale_factor,
                      mode = self$mode)
    }
  )
  mod(size, scale_factor, mode)
}
