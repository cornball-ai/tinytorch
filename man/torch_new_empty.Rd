% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_new_empty}
\alias{torch_new_empty}
\title{New empty}
\usage{
torch_new_empty(self, size, dtype = NULL, device = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::new_empty}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
