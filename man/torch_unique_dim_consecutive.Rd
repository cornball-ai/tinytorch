% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unique_dim_consecutive}
\alias{torch_unique_dim_consecutive}
\title{Unique dim consecutive}
\usage{
torch_unique_dim_consecutive(self, dim, return_inverse = FALSE,
                             return_counts = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{return_inverse}{A logical value.}

\item{return_counts}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unique_dim_consecutive}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
