% tinyrox says don't edit this manually, but it can't stop you!
\name{constant_fold}
\alias{constant_fold}
\title{Constant Folding}
\usage{
constant_fold(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with constants folded
}
\description{
If all inputs to a node are constants, evaluate at compile time
and replace the node with a constant.
}
