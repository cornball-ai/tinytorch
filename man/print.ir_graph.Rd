% tinyrox says don't edit this manually, but it can't stop you!
\name{print.ir_graph}
\alias{print.ir_graph}
\title{Print an IR Graph}
\usage{
\method{print}{ir_graph}(x, ...)
}
\arguments{
\item{x}{An ir_graph}

\item{...}{Ignored}
}
\value{
Invisibly returns x
}
\description{
Print an IR Graph
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$relu()))
e <- new.env()
e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
print(g)
}
}
