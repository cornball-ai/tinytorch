% tinyrox says don't edit this manually, but it can't stop you!
\name{plan_memory}
\alias{plan_memory}
\title{Plan Memory Allocation for an IR Graph}
\usage{
plan_memory(graph, input_shapes = NULL, input_dtypes = NULL)
}
\arguments{
\item{graph}{An ir_graph}

\item{input_shapes}{Named list of input shapes (used if graph lacks annotations)}

\item{input_dtypes}{Optional named list of input dtypes}
}
\value{
A memory_plan object
}
\description{
Uses shape inference and liveness analysis to pre-plan all memory
allocations. Applies greedy interval scheduling (best-fit) for
buffer reuse and detects in-place operation opportunities.
}
\examples{
\donttest{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
plan_memory(g, list(x = c(2L, 3L)))
}
}
