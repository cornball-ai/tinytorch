% tinyrox says don't edit this manually, but it can't stop you!
\name{validate_ir}
\alias{validate_ir}
\title{Validate an IR Graph}
\usage{
validate_ir(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
TRUE if valid, otherwise stops with an error
}
\description{
Checks that all input references resolve, there are no cycles,
and nodes are in topological order.
}
