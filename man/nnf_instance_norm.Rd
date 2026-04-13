% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_instance_norm}
\alias{nnf_instance_norm}
\title{Nnf instance norm}
\usage{
nnf_instance_norm(input, weight, bias, running_mean, running_var,
                  use_input_stats, momentum, eps, cudnn_enabled)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{running_mean}{A `torch_tensor` of running means.}

\item{running_var}{A `torch_tensor` of running variances.}

\item{use_input_stats}{Parameter passed to the underlying ATen operator.}

\item{momentum}{Numeric; momentum value.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{cudnn_enabled}{Logical.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf instance norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
