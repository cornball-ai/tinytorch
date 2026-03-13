% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_max_pool2d}
\alias{torch_max_pool2d}
\title{Max pool2d}
\usage{
torch_max_pool2d(self, kernel_size, stride = NULL, padding = 0, dilation = 1,
                 ceil_mode = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{stride}{(int[2]).}

\item{padding}{(int[2]).}

\item{dilation}{(int[2]).}

\item{ceil_mode}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::max_pool2d}.
}
