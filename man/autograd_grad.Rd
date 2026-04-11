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
\examples{
\donttest{
if (torch_is_installed()) {
w <- torch_tensor(0.5, requires_grad = TRUE)
b <- torch_tensor(0.9, requires_grad = TRUE)
x <- torch_tensor(runif(100))
y <- 2 * x + 1
loss <- (y - (w * x + b))^2
loss <- loss$mean()

o <- autograd_grad(loss, list(w, b))
o
}
}
}
