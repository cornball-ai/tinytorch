% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_log_normal_}
\alias{torch_log_normal_}
\title{Log normal}
\usage{
torch_log_normal_(self, mean = 1, std = 2, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mean}{A numeric value.}

\item{std}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::log_normal_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
