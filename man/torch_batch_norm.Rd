% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_batch_norm}
\alias{torch_batch_norm}
\title{torch_batch_norm}
\usage{
torch_batch_norm(input, weight = NULL, bias = NULL, running_mean = NULL,
                 running_var = NULL, training = FALSE, momentum = 0.1,
                 eps = 1e-05, cudnn_enabled = TRUE)
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
torch_batch_norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
