% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_householder_product}
\alias{torch_linalg_householder_product}
\alias{linalg_householder_product}
\title{Linalg householder product}
\usage{
torch_linalg_householder_product(input, tau)
}
\arguments{
\item{input}{A torch_tensor.}

\item{tau}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_householder_product}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
