% tinyrox says don't edit this manually, but it can't stop you!
\name{common_subexpr_eliminate}
\alias{common_subexpr_eliminate}
\title{Common Subexpression Elimination}
\usage{
common_subexpr_eliminate(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with duplicate expressions eliminated
}
\description{
If two nodes have the same op, same inputs, and same attrs,
replace the second with a reference to the first.
}
