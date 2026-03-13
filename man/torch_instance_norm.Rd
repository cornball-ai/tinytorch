% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_instance_norm}
\alias{torch_instance_norm}
\title{Instance norm}
\usage{
torch_instance_norm(input, weight, bias, running_mean, running_var,
                    use_input_stats, momentum, eps, cudnn_enabled)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{running_mean}{A torch_tensor.}

\item{running_var}{A torch_tensor.}

\item{use_input_stats}{A logical value.}

\item{momentum}{A numeric value.}

\item{eps}{A numeric value.}

\item{cudnn_enabled}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::instance_norm}.
}
