% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantized_batch_norm}
\alias{torch_quantized_batch_norm}
\title{Quantized batch norm}
\usage{
torch_quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale,
                           output_zero_point)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{mean}{A torch_tensor.}

\item{var}{A torch_tensor.}

\item{eps}{A numeric value.}

\item{output_scale}{A numeric value.}

\item{output_zero_point}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantized_batch_norm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
