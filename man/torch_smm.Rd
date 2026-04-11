% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_smm}
\alias{torch_smm}
\title{Smm}
\usage{
torch_smm(self, mat2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mat2}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::smm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
