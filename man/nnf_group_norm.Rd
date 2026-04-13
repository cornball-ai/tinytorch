% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_group_norm}
\alias{nnf_group_norm}
\title{Nnf group norm}
\usage{
nnf_group_norm(input, num_groups, weight, bias, eps, cudnn_enabled)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{num_groups}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{cudnn_enabled}{Logical.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf group norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
