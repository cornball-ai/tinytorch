% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm_gather_stats_with_counts}
\alias{torch_batch_norm_gather_stats_with_counts}
\title{Batch norm gather stats with counts}
\usage{
torch_batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean,
                                          running_var, momentum, eps, counts)
}
\arguments{
\item{input}{A torch_tensor.}

\item{mean}{A torch_tensor.}

\item{invstd}{A torch_tensor.}

\item{running_mean}{A torch_tensor.}

\item{running_var}{A torch_tensor.}

\item{momentum}{A numeric value.}

\item{eps}{A numeric value.}

\item{counts}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::batch_norm_gather_stats_with_counts}.
}
