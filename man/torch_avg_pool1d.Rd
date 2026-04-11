% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_avg_pool1d}
\alias{torch_avg_pool1d}
\title{Avg pool1d}
\usage{
torch_avg_pool1d(self, kernel_size, stride = NULL, padding = 0,
                 ceil_mode = FALSE, count_include_pad = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[1]).}

\item{stride}{(int[1]).}

\item{padding}{(int[1]).}

\item{ceil_mode}{A logical value.}

\item{count_include_pad}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::avg_pool1d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
