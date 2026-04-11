% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rename}
\alias{torch_rename}
\title{Rename}
\usage{
torch_rename(self, names)
}
\arguments{
\item{self}{A torch_tensor.}

\item{names}{A character vector of dimension names.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rename}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
