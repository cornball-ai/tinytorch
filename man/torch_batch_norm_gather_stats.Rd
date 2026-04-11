% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm_gather_stats}
\alias{torch_batch_norm_gather_stats}
\title{Batch norm gather stats}
\usage{
torch_batch_norm_gather_stats(input, mean, invstd, running_mean, running_var,
                              momentum, eps, count)
}
\arguments{
\item{input}{A torch_tensor.}

\item{mean}{A torch_tensor.}

\item{invstd}{A torch_tensor.}

\item{running_mean}{A torch_tensor.}

\item{running_var}{A torch_tensor.}

\item{momentum}{A numeric value.}

\item{eps}{A numeric value.}

\item{count}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::batch_norm_gather_stats}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
