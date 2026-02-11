% tinyrox says don't edit this manually, but it can't stop you!
\name{is_graph_safe}
\alias{is_graph_safe}
\title{Check if an Expression is Graph-Safe}
\usage{
is_graph_safe(expr)
}
\arguments{
\item{expr}{An R expression or a walked node tree}
}
\value{
Logical, TRUE if entirely graph-safe
}
\description{
Walks the expression tree and checks if all nodes are graph-safe
(can be compiled to a torch graph without breaks).
}
\examples{
is_graph_safe(quote(y <- x$matmul(w)))
is_graph_safe(quote(print(x)))
}
