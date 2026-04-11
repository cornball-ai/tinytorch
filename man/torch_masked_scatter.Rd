% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_masked_scatter}
\alias{torch_masked_scatter}
\title{Masked scatter}
\usage{
torch_masked_scatter(self, mask, source)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mask}{A torch_tensor.}

\item{source}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::masked_scatter}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
