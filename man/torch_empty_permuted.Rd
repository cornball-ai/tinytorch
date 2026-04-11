% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_empty_permuted}
\alias{torch_empty_permuted}
\title{Empty permuted}
\usage{
torch_empty_permuted(size, physical_layout, dtype = NULL, device = NULL)
}
\arguments{
\item{size}{An integer vector.}

\item{physical_layout}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::empty_permuted}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
