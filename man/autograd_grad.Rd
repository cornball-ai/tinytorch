% tinyrox says don't edit this manually, but it can't stop you!
\name{autograd_grad}
\alias{autograd_grad}
\title{Compute gradients}
\usage{
autograd_grad(outputs, inputs, grad_outputs = NULL, retain_graph = FALSE,
              create_graph = FALSE)
}
\arguments{
\item{outputs}{Tensor(s) to differentiate.}

\item{inputs}{Tensor(s) to differentiate with respect to.}

\item{grad_outputs}{Gradient of outputs.}

\item{retain_graph}{Keep computation graph.}

\item{create_graph}{Create graph for higher-order gradients.}
}
\value{
List of gradient tensors.
}
\description{
Compute gradients
}
