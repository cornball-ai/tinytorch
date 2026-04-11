% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_special_polygamma}
\alias{torch_special_polygamma}
\title{Special polygamma}
\usage{
torch_special_polygamma(n, self)
}
\arguments{
\item{n}{An integer.}

\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::special_polygamma}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
