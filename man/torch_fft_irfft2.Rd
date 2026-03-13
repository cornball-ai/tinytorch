% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_irfft2}
\alias{torch_fft_irfft2}
\title{Fft irfft2}
\usage{
torch_fft_irfft2(self, s = NULL, dim = NULL, norm = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{s}{(int[1]). Optional.}

\item{dim}{(int[1]).}

\item{norm}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_irfft2}.
}
