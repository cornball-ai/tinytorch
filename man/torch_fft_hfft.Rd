% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_hfft}
\alias{torch_fft_hfft}
\title{Fft hfft}
\usage{
torch_fft_hfft(self, n = NULL, dim = -1, norm = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{n}{An integer. Optional.}

\item{dim}{An integer.}

\item{norm}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_hfft}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
