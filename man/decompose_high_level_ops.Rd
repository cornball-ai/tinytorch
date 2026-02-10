% tinyrox says don't edit this manually, but it can't stop you!
\name{decompose_high_level_ops}
\alias{decompose_high_level_ops}
\title{Decompose High-Level Ops to Primitives}
\usage{
decompose_high_level_ops(graph)
}
\arguments{
\item{graph}{An ir_graph}
}
\value{
A new ir_graph with decomposed ops
}
\description{
Rewrites composite ops into lower-level primitives to expose
fuseable elementwise chains:
\itemize{
\item \code{torch_linear(x, w, b)} -> \code{transpose(w) + matmul(x, wt) + add(result, b)}
\item \code{torch_gelu(x)} -> \code{gelu(x)}
\item \code{torch_silu(x)} -> \code{silu(x)}
\item \code{torch_sigmoid(x)} -> \code{sigmoid(x)}
}
\code{torch_layer_norm} and \code{torch_conv1d} are kept as-is
(they contain reductions or special layouts that can't be
elementwise-fused).
}
