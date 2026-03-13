% tinyrox says don't edit this manually, but it can't stop you!
\name{fusion_annotate}
\alias{fusion_annotate}
\title{Fusion Annotation}
\usage{
fusion_annotate(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with fusion_group annotations
}
\description{
Identifies maximal fuseable subgraphs: contiguous chains of
elementwise operations where each intermediate result has
exactly one consumer. Annotates nodes with a \code{fusion_group} ID.
}
\examples{
\dontrun{
stmts <- list(quote(y <- x$relu()$sigmoid()))
e <- new.env(); e$x <- torch_randn(c(2, 3))
g <- lower_to_ir(stmts, e)
fusion_annotate(g)
}
}
