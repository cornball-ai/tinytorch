% tinyrox says don't edit this manually, but it can't stop you!
\name{dead_code_eliminate}
\alias{dead_code_eliminate}
\title{Dead Code Elimination}
\usage{
dead_code_eliminate(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with dead nodes removed
}
\description{
Remove nodes whose outputs are never consumed.
Walks backward from outputs to find reachable nodes.
}
