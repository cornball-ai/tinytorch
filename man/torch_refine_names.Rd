% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_refine_names}
\alias{torch_refine_names}
\title{Refine names}
\usage{
torch_refine_names(self, names)
}
\arguments{
\item{self}{A torch_tensor.}

\item{names}{A character vector of dimension names.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::refine_names}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
