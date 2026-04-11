% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_layer_norm}
\alias{torch_native_layer_norm}
\title{Native layer norm}
\usage{
torch_native_layer_norm(input, normalized_shape, weight, bias, eps)
}
\arguments{
\item{input}{A torch_tensor.}

\item{normalized_shape}{An integer vector.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_layer_norm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
