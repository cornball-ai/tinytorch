% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_put_}
\alias{torch_index_put_}
\title{Index put}
\usage{
torch_index_put_(self, indices, values, accumulate = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{indices}{(Tensor?[]).}

\item{values}{A torch_tensor.}

\item{accumulate}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::index_put_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
