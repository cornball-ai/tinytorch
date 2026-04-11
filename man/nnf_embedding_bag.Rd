% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_embedding_bag}
\alias{nnf_embedding_bag}
\title{Nnf embedding bag}
\usage{
nnf_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                  per_sample_weights, include_last_offset)
}
\arguments{
\item{weight}{A `torch_tensor` of weights.}

\item{indices}{A `torch_tensor` of indices.}

\item{offsets}{Parameter passed to the underlying ATen operator.}

\item{scale_grad_by_freq}{Parameter passed to the underlying ATen operator.}

\item{mode}{Parameter passed to the underlying ATen operator.}

\item{sparse}{Parameter passed to the underlying ATen operator.}

\item{per_sample_weights}{Parameter passed to the underlying ATen operator.}

\item{include_last_offset}{Parameter passed to the underlying ATen operator.}
}
\description{
Nnf embedding bag
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
