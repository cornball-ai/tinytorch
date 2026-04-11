% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_col2im}
\alias{torch_col2im}
\title{Col2im}
\usage{
torch_col2im(self, output_size, kernel_size, dilation, padding, stride)
}
\arguments{
\item{self}{A torch_tensor.}

\item{output_size}{(int[2]).}

\item{kernel_size}{(int[2]).}

\item{dilation}{(int[2]).}

\item{padding}{(int[2]).}

\item{stride}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::col2im}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
