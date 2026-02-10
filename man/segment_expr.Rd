% tinyrox says don't edit this manually, but it can't stop you!
\name{segment_expr}
\alias{segment_expr}
\title{Graph Segmentation}
\usage{
segment_expr(expr)
}
\arguments{
\item{expr}{An R expression (typically a block `\{...\}`)}
}
\value{
A list of segments, each with `type` ("graph" or "r_code")
  and `statements` (list of expressions)
}
\description{
Splits an expression at graph break points into segments that can
each be compiled independently, with R code executed between them.
Segment an Expression at Graph Breaks
Takes a block expression and splits it into segments:
- "graph" segments contain consecutive graph-safe statements
- "r_code" segments contain graph-breaking statements
}
