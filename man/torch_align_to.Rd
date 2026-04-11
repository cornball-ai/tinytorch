% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_align_to}
\alias{torch_align_to}
\title{Align to}
\usage{
torch_align_to(self, names)
}
\arguments{
\item{self}{A torch_tensor.}

\item{names}{A character vector of dimension names.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::align_to}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
