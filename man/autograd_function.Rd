% tinyrox says don't edit this manually, but it can't stop you!
\name{autograd_function}
\alias{autograd_function}
\title{Define a custom autograd function}
\usage{
autograd_function(forward, backward)
}
\arguments{
\item{forward}{Forward function.}

\item{backward}{Backward function.}
}
\value{
An autograd function class.
}
\description{
Define a custom autograd function
}
\examples{
\donttest{
if (torch_is_installed()) {

exp2 <- autograd_function(
  forward = function(ctx, i) {
    result <- i$exp()
    ctx$save_for_backward(result = result)
    result
  },
  backward = function(ctx, grad_output) {
    list(i = grad_output * ctx$saved_variable$result)
  }
)
}
}
}
