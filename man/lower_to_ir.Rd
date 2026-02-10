% tinyrox says don't edit this manually, but it can't stop you!
\name{lower_to_ir}
\alias{lower_to_ir}
\title{Lower R AST Statements to IR}
\usage{
lower_to_ir(statements, env = parent.frame())
}
\arguments{
\item{statements}{List of R expressions (from segment_expr())}

\item{env}{Environment with tensor values}
}
\value{
An ir_graph
}
\description{
Takes a graph-safe segment's statements (raw R expressions) plus
the evaluation environment. Walks each statement recursively,
emitting IR nodes.
}
