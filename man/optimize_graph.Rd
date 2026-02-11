% tinyrox says don't edit this manually, but it can't stop you!
\name{optimize_graph}
\alias{optimize_graph}
\title{Run Optimization Pipeline}
\usage{
optimize_graph(graph, passes = NULL)
}
\arguments{
\item{graph}{An ir_graph}

\item{passes}{List of pass functions. If NULL, uses default pipeline.}
}
\value{
An optimized ir_graph
}
\description{
Applies a sequence of optimization passes to an IR graph.
Default pipeline: constant_fold, dead_code_eliminate,
common_subexpr_eliminate, algebraic_simplify, dead_code_eliminate,
fusion_annotate.
}
\examples{
\donttest{
stmts <- list(quote(y <- x$relu()$sigmoid()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
optimize_graph(g)
}
}
