% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rename_}
\alias{torch_rename_}
\title{Rename}
\usage{
torch_rename_(self, names)
}
\arguments{
\item{self}{A torch_tensor.}

\item{names}{A character vector of dimension names.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rename_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
