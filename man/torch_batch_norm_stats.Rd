% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm_stats}
\alias{torch_batch_norm_stats}
\title{Batch norm stats}
\usage{
torch_batch_norm_stats(input, eps)
}
\arguments{
\item{input}{A torch_tensor.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::batch_norm_stats}.
}
