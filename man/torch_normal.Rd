% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_normal}
\alias{torch_normal}
\title{Normal}
\usage{
torch_normal(mean, std = 1, generator = NULL)
}
\arguments{
\item{mean}{A torch_tensor.}

\item{std}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::normal}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
