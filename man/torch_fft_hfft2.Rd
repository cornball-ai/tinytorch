% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_hfft2}
\alias{torch_fft_hfft2}
\title{Fft hfft2}
\usage{
torch_fft_hfft2(self, s = NULL, dim = NULL, norm = NULL)
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
Wrapper for \code{at::fft_hfft2}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
