% tinyrox says don't edit this manually, but it can't stop you!
\name{walk_expr}
\alias{walk_expr}
\title{Walk an Expression and Build a Node Tree}
\usage{
walk_expr(expr, depth = 0L)
}
\arguments{
\item{expr}{An R expression}

\item{depth}{Current recursion depth (for debugging)}
}
\value{
A nested list representing the classified AST
}
\description{
Recursively walks an R expression, classifying each node.
}
