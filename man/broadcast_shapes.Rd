% tinyrox says don't edit this manually, but it can't stop you!
\name{broadcast_shapes}
\alias{broadcast_shapes}
\title{Analysis Passes for IR Graphs}
\usage{
broadcast_shapes(shape_a, shape_b)
}
\arguments{
\item{shape_a}{Integer vector}

\item{shape_b}{Integer vector}
}
\value{
Integer vector of broadcast shape, or NULL if incompatible
}
\description{
Shape inference, liveness analysis, and dependency analysis.
These annotate IR graphs with information needed by optimization passes.
Broadcast Two Shapes (NumPy-compatible)
Right-aligns dimensions and checks compatibility.
Each dimension pair must be equal, or one must be 1.
}
\examples{
broadcast_shapes(c(2, 1), c(1, 3))
}
