% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_ifft}
\alias{torch_fft_ifft}
\title{Fft ifft}
\usage{
torch_fft_ifft(self, n = NULL, dim = -1, norm = NULL)
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
Wrapper for \code{at::fft_ifft}.
}
\examples{
\donttest{
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
t
x <- torch_fft_fft(t, norm = "backward")
torch_fft_ifft(x)


}
}
}
