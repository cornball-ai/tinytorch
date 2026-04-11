% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_cosine_similarity}
\alias{nnf_cosine_similarity}
\title{Nnf cosine similarity}
\usage{
nnf_cosine_similarity(x1, x2, dim, eps)
}
\arguments{
\item{x1}{Parameter passed to the underlying ATen operator.}

\item{x2}{Parameter passed to the underlying ATen operator.}

\item{dim}{Integer dimension.}

\item{eps}{Numeric; small constant for numerical stability.}
}
\description{
Nnf cosine similarity
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
