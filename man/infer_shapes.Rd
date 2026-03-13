% tinyrox says don't edit this manually, but it can't stop you!
\name{infer_shapes}
\alias{infer_shapes}
\title{Infer Shapes and Dtypes Through an IR Graph}
\usage{
infer_shapes(graph, input_shapes, input_dtypes = NULL)
}
\arguments{
\item{graph}{An ir_graph}

\item{input_shapes}{Named list mapping input names to integer vectors}

\item{input_dtypes}{Optional named list mapping input names to dtype strings}
}
\value{
A new ir_graph with shape/dtype annotations
}
\description{
Forward pass: propagates shapes and dtypes from inputs through every node.
Returns a new ir_graph with \code{output_shape} and \code{output_dtype}
in each node's attrs.
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
infer_shapes(g, list(x = c(2L, 3L)))
}
}
