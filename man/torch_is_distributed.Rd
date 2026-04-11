% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_distributed}
\alias{torch_is_distributed}
\title{Is distributed}
\usage{
torch_is_distributed(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_distributed}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
