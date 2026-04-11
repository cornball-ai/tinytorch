% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_fft}
\alias{torch_fft_fft}
\title{Fft fft}
\usage{
torch_fft_fft(self, n = NULL, dim = -1, norm = NULL)
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
Wrapper for \code{at::fft_fft}.
}
\examples{
\donttest{
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
t
torch_fft_fft(t, norm = "backward")

}
}
}
