% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_embedding_bag}
\alias{torch_embedding_bag}
\title{Embedding bag}
\usage{
torch_embedding_bag(weight, indices, offsets, scale_grad_by_freq = FALSE,
                    mode = 0, sparse = FALSE, per_sample_weights = NULL,
                    include_last_offset = FALSE)
}
\arguments{
\item{weight}{A torch_tensor.}

\item{indices}{A torch_tensor.}

\item{offsets}{A torch_tensor.}

\item{scale_grad_by_freq}{A logical value.}

\item{mode}{An integer.}

\item{sparse}{A logical value.}

\item{per_sample_weights}{A torch_tensor. Optional.}

\item{include_last_offset}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::embedding_bag}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
