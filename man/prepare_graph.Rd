% tinyrox says don't edit this manually, but it can't stop you!
\name{prepare_graph}
\alias{prepare_graph}
\title{Prepare an IR Graph for Fast Execution}
\usage{
prepare_graph(graph, example_inputs, optimize = TRUE, fuse = TRUE,
              fuse_matmul_epilogues = FALSE, backend = "auto")
}
\arguments{
\item{graph}{An ir_graph}

\item{example_inputs}{Named list of input torch_tensors (used for
shape inference and fusion compilation)}

\item{optimize}{Logical, run optimization passes (default TRUE)}

\item{fuse}{Logical, compile fusion groups to kernels (default TRUE)}

\item{fuse_matmul_epilogues}{Logical, fuse matmul+bias+activation on GPU (default FALSE)}

\item{backend}{Character: "auto", "gpu", or "cpu". "auto" detects
from input tensors.}
}
\value{
A \code{prepared_graph} object
}
\description{
Performs all expensive optimization, shape inference, and fusion
compilation once. Returns a prepared execution plan that can be
executed many times via \code{execute_prepared()}.
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
pg <- prepare_graph(g, list(x = torch_randn(c(2, 3))))
}
}
