% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_max_pool1d_with_indices}
\alias{torch_max_pool1d_with_indices}
\title{Max pool1d with indices}
\usage{
torch_max_pool1d_with_indices(self, kernel_size, stride = NULL, padding = 0,
                              dilation = 1, ceil_mode = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[1]).}

\item{stride}{(int[1]).}

\item{padding}{(int[1]).}

\item{dilation}{(int[1]).}

\item{ceil_mode}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::max_pool1d_with_indices}.
}
