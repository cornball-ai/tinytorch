% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_vsplit}
\alias{torch_vsplit}
\title{Vsplit}
\usage{
torch_vsplit(self, sections)
}
\arguments{
\item{self}{A torch_tensor.}

\item{sections}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::vsplit}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
