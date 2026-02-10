% tinyrox says don't edit this manually, but it can't stop you!
\name{analyze_deps}
\alias{analyze_deps}
\title{Dependency Analysis}
\usage{
analyze_deps(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
Named list of node_id (as character) to integer vector of dependency IDs
}
\description{
For each node, compute the full transitive dependency set.
}
