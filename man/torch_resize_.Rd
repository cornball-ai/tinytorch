% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_resize_}
\alias{torch_resize_}
\title{Resize}
\usage{
torch_resize_(self, size, memory_format = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{memory_format}{A memory format. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::resize_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
