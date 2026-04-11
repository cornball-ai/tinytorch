% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_batch_norm}
\alias{nnf_batch_norm}
\title{Nnf batch norm}
\usage{
nnf_batch_norm(input, weight, bias, running_mean, running_var, training,
               momentum, eps, cudnn_enabled)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{running_mean}{A `torch_tensor` of running means.}

\item{running_var}{A `torch_tensor` of running variances.}

\item{training}{Logical.}

\item{momentum}{Numeric; momentum value.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{cudnn_enabled}{Logical.}
}
\description{
Nnf batch norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
