% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_empty_strided}
\alias{torch_empty_strided}
\title{Empty strided}
\usage{
torch_empty_strided(size, stride, dtype = NULL, device = NULL)
}
\arguments{
\item{size}{An integer vector.}

\item{stride}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::empty_strided}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
