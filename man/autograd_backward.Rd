% tinyrox says don't edit this manually, but it can't stop you!
\name{autograd_backward}
\alias{autograd_backward}
\title{Autograd backward}
\usage{
autograd_backward(tensors, grad_tensors = NULL)
}
\arguments{
\item{tensors}{Parameter passed to the underlying ATen operator.}

\item{grad_tensors}{Parameter passed to the underlying ATen operator.}
}
\description{
Autograd backward
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
