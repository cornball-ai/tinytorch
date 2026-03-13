% tinyrox says don't edit this manually, but it can't stop you!
\name{execute_prepared}
\alias{execute_prepared}
\title{Execute a Prepared Graph}
\usage{
execute_prepared(prepared, inputs, verbose = FALSE)
}
\arguments{
\item{prepared}{A \code{prepared_graph} object from \code{prepare_graph()}}

\item{inputs}{Named list of input torch_tensors}

\item{verbose}{Logical, print execution info}
}
\value{
A torch_tensor (or list of tensors for multi-output graphs)
}
\description{
Fast execution path using a pre-computed execution plan from
\code{prepare_graph()}. No optimization or compilation overhead --
just runs the execution loop.
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$relu()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
pg <- prepare_graph(g, list(x = torch_randn(c(2, 3))))
execute_prepared(pg, list(x = torch_randn(c(2, 3))))
}
}
