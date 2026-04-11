% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_fftshift}
\alias{torch_fft_fftshift}
\title{Fft fftshift}
\usage{
torch_fft_fftshift(self, dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_fftshift}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
