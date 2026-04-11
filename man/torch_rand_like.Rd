% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rand_like}
\alias{torch_rand_like}
\title{Rand like}
\usage{
torch_rand_like(self, dtype = NULL, device = NULL, memory_format = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}

\item{memory_format}{A memory format. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rand_like}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
