% tinyrox says don't edit this manually, but it can't stop you!
\name{execute_optimized}
\alias{execute_optimized}
\title{Execute an Optimized IR Graph}
\usage{
execute_optimized(
  graph,
  inputs,
  optimize = TRUE,
  fuse = TRUE,
  fuse_matmul_epilogues = FALSE,
  backend = "auto",
  verbose = FALSE
)
}
\arguments{
\item{graph}{An ir_graph}

\item{inputs}{Named list of input torch_tensors}

\item{optimize}{Logical, run optimization passes (default TRUE)}

\item{fuse}{Logical, compile fusion groups to kernels (default TRUE)}

\item{fuse_matmul_epilogues}{Logical, fuse matmul+bias+activation on GPU (default FALSE)}

\item{backend}{Character: "auto", "gpu", or "cpu". "auto" detects
from input tensors and ariel availability.}

\item{verbose}{Logical, print execution info}
}
\value{
A torch_tensor (or list of tensors for multi-output graphs)
}
\description{
Runs the full optimization pipeline (constant folding, DCE, CSE,
algebraic simplification, fusion annotation) then executes the graph.
Fusion groups are compiled to SIMD-vectorized C++ kernels (CPU) or
Triton GPU kernels (via ariel); non-fused ops are dispatched to torch.
Automatically caches prepared graphs keyed by graph structure,
input shapes, and backend. For single-op graphs, uses eager fallback
to avoid all pipeline overhead.
For maximum performance in tight loops, use \code{prepare_graph()}
and \code{execute_prepared()} directly.
}
\examples{
\donttest{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
execute_optimized(g, list(x = torch_randn(c(2, 3))))
}
}
