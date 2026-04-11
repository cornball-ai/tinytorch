% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_embedding}
\alias{torch_embedding}
\title{Embedding lookup}
\usage{
torch_embedding(weight, indices)
}
\arguments{
\item{weight}{Embedding matrix tensor.}

\item{indices}{Integer tensor of indices.}
}
\description{
Embedding lookup
}
\examples{
\donttest{
if (torch_is_installed()) {
emb <- torch_randn(c(10, 4))
idx <- torch_tensor(c(1L, 3L, 5L))
torch_embedding(emb, idx)
}
}
}
