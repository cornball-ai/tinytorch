% tinyrox says don't edit this manually, but it can't stop you!
\name{analyze_liveness}
\alias{analyze_liveness}
\title{Liveness Analysis}
\usage{
analyze_liveness(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
Named list of node_id (as character) to last_use_id (integer or Inf)
}
\description{
Backward pass: for each node, compute the last point its output is consumed.
Nodes in \code{output_ids} have \code{last_use = Inf} (live at graph exit).
Nodes with no consumers get \code{last_use = NA} (dead code).
}
\examples{
\donttest{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
analyze_liveness(g)
}
}
