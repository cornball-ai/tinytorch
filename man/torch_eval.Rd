% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_eval}
\alias{torch_eval}
\title{Evaluate R Expressions as Torch Computations}
\usage{
torch_eval(
  expr,
  ...,
  .env = parent.frame(),
  .compile = FALSE,
  .fuse = TRUE,
  .verbose = FALSE
)
}
\arguments{
\item{expr}{An R expression involving torch tensor operations.
Captured unevaluated via NSE.}

\item{...}{Named torch tensors to make available in the expression.}

\item{.env}{Environment for resolving non-tensor symbols.}

\item{.compile}{Logical. If TRUE, attempts to compile graph-safe
segments using jit_trace. Default FALSE (direct execution).}

\item{.fuse}{Logical. If TRUE, detects fuseable patterns and
dispatches to pre-compiled SIMD kernels. Default TRUE.}

\item{.verbose}{Logical. If TRUE, prints compilation info.}
}
\value{
The result of evaluating the expression (typically a torch tensor).
}
\description{
Captures an R expression and evaluates it, optionally compiling
graph-safe segments using jit_trace or fused SIMD kernels.
}
\examples{
\dontrun{
x <- torch_randn(3, 3)
y <- torch_randn(3, 3)

# Direct execution with automatic fusion (default)
result <- torch_eval({
  z <- x$matmul(y)
  z$relu()$sigmoid()
}, x = x, y = y)

# With jit_trace compilation
result <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x, y = y, .compile = TRUE)
}

}
