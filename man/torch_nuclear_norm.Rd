% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nuclear_norm}
\alias{torch_nuclear_norm}
\title{Nuclear norm}
\usage{
torch_nuclear_norm(self, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nuclear_norm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
