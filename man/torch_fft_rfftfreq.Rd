% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_rfftfreq}
\alias{torch_fft_rfftfreq}
\title{Fft rfftfreq}
\usage{
torch_fft_rfftfreq(n, d = 1, dtype = NULL, device = NULL)
}
\arguments{
\item{n}{An integer.}

\item{d}{A numeric value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_rfftfreq}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
