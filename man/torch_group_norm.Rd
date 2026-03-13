% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_group_norm}
\alias{torch_group_norm}
\title{Group norm}
\usage{
torch_group_norm(input, num_groups, weight = NULL, bias = NULL, eps = 1e-05,
                 cudnn_enabled = TRUE)
}
\arguments{
\item{input}{A torch_tensor.}

\item{num_groups}{An integer.}

\item{weight}{A torch_tensor. Optional.}

\item{bias}{A torch_tensor. Optional.}

\item{eps}{A numeric value.}

\item{cudnn_enabled}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::group_norm}.
}
