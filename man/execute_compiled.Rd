% tinyrox says don't edit this manually, but it can't stop you!
\name{execute_compiled}
\alias{execute_compiled}
\title{Execute Expression with Compilation}
\usage{
execute_compiled(expr, env, compile = TRUE, fuse = TRUE, verbose = FALSE)
}
\arguments{
\item{expr}{An R expression}

\item{env}{Environment with tensors}

\item{compile}{Logical, whether to use jit_trace}

\item{fuse}{Logical, whether to try fused kernel dispatch}

\item{verbose}{Logical, whether to print compilation info}
}
\value{
The result of the expression
}
\description{
Segments an expression, compiles graph-safe segments, and executes.
Supports two optimization strategies:
- **Fusion** (.fuse): Pattern-matches operations and dispatches to
pre-compiled SIMD kernels. Fast, no compilation overhead.
- **Compilation** (.compile): Traces segments with jit_trace.
Better for long chains but has trace overhead.
Fusion is tried first. Operations that don't match a fused pattern
fall through to jit_trace (if enabled) or direct execution.
}
\examples{
\dontrun{
x <- torch_randn(c(2, 3))
env <- new.env()
env$x <- x
execute_compiled(quote(x$relu()), env)
}
}
