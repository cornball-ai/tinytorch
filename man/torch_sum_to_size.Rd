% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sum_to_size}
\alias{torch_sum_to_size}
\title{Sum to size}
\usage{
torch_sum_to_size(self, size)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sum_to_size}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
