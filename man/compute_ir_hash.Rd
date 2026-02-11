% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_ir_hash}
\alias{compute_ir_hash}
\title{Compute IR hash for cache key}
\usage{
compute_ir_hash(graph)
}
\arguments{
\item{graph}{IR graph object}
}
\value{
Character hash string
}
\description{
Compute IR hash for cache key
}
\examples{
\donttest{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
compute_ir_hash(g)
}
}
