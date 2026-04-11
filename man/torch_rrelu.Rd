% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rrelu}
\alias{torch_rrelu}
\title{Rrelu}
\usage{
torch_rrelu(self, lower = 0.125, upper = 0.333333333333333, training = FALSE,
            generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{lower}{A numeric scalar.}

\item{upper}{A numeric scalar.}

\item{training}{A logical value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rrelu}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
