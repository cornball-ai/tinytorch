% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_coalesce}
\alias{torch_coalesce}
\title{Coalesce}
\usage{
torch_coalesce(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::coalesce}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
