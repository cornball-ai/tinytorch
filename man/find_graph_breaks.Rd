% tinyrox says don't edit this manually, but it can't stop you!
\name{find_graph_breaks}
\alias{find_graph_breaks}
\title{Find Graph Breaks in an Expression}
\usage{
find_graph_breaks(expr)
}
\arguments{
\item{expr}{An R expression}
}
\value{
A list of graph-breaking nodes with their details
}
\description{
Identifies nodes that cause graph breaks.
}
\examples{
find_graph_breaks(quote({ y <- x$relu(); print(y) }))
}
