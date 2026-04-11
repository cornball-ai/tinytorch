% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_select_scatter}
\alias{torch_select_scatter}
\title{Select scatter}
\usage{
torch_select_scatter(self, src, dim, index)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::select_scatter}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
