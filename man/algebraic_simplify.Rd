% tinyrox says don't edit this manually, but it can't stop you!
\name{algebraic_simplify}
\alias{algebraic_simplify}
\title{Algebraic Simplification}
\usage{
algebraic_simplify(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with algebraic simplifications applied
}
\description{
Pattern-match and rewrite known algebraic identities.
Assumes fast-math semantics (no NaN/Inf guards).
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$mul(torch_tensor(1.0))))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
algebraic_simplify(g)
}
}
