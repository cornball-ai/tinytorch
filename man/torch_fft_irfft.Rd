% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_irfft}
\alias{torch_fft_irfft}
\title{Fft irfft}
\usage{
torch_fft_irfft(self, n = NULL, dim = -1, norm = NULL)
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
Wrapper for \code{at::fft_irfft}.
}
\examples{
\donttest{
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 4)
x <- torch_fft_rfft(t)
torch_fft_irfft(x)
torch_fft_irfft(x, n = t$numel())

}
}
}
