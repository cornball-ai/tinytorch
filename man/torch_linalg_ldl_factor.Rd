% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_ldl_factor}
\alias{torch_linalg_ldl_factor}
\alias{linalg_ldl_factor}
\title{Linalg ldl factor}
\usage{
torch_linalg_ldl_factor(self, hermitian = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{hermitian}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_ldl_factor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
