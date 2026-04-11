% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm_update_stats}
\alias{torch_batch_norm_update_stats}
\title{Batch norm update stats}
\usage{
torch_batch_norm_update_stats(input, running_mean, running_var, momentum)
}
\arguments{
\item{input}{A torch_tensor.}

\item{running_mean}{A torch_tensor.}

\item{running_var}{A torch_tensor.}

\item{momentum}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::batch_norm_update_stats}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
