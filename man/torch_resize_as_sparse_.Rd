% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_resize_as_sparse_}
\alias{torch_resize_as_sparse_}
\title{Resize as sparse}
\usage{
torch_resize_as_sparse_(self, the_template)
}
\arguments{
\item{self}{A torch_tensor.}

\item{the_template}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::resize_as_sparse_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
