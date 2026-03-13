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
