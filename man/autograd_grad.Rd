% tinyrox says don't edit this manually, but it can't stop you!
\name{autograd_grad}
\alias{autograd_grad}
\title{Compute gradients}
\usage{
autograd_grad(outputs, inputs, grad_outputs = NULL, retain_graph = FALSE,
              create_graph = FALSE, allow_unused = FALSE)
}
\arguments{
\item{outputs}{A `torch_tensor` or list of tensors to differentiate.}

\item{inputs}{A `torch_tensor` or list of tensors to differentiate with
respect to.}

\item{grad_outputs}{Optional list of gradients of outputs; defaults to
ones for scalar outputs.}

\item{retain_graph}{Logical; if `TRUE` the computation graph is retained
for a subsequent backward pass.}

\item{create_graph}{Logical; if `TRUE` construct a graph over the
gradients themselves so higher-order derivatives can be taken.}

\item{allow_unused}{Logical; if `TRUE` inputs that do not contribute to
outputs get `NULL` gradients instead of raising.}
}
\value{
A list of gradient tensors, one per input.
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
loss <- ((y - (w * x + b))^2)$mean()

o <- autograd_grad(loss, list(w, b))
o
}
}
}
