% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm_elemt}
\alias{torch_batch_norm_elemt}
\title{Batch norm elemt}
\usage{
torch_batch_norm_elemt(input, weight, bias, mean, invstd, eps)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{mean}{A torch_tensor.}

\item{invstd}{A torch_tensor.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::batch_norm_elemt}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
