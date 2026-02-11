% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_module}
\alias{nn_module}
\title{Create an nn_module class}
\usage{
nn_module(classname = NULL, ...)
}
\arguments{
\item{classname}{Character name for the module class (optional,
defaults to "nn_module").}

\item{...}{Named functions: initialize, forward, and any other methods.}
}
\value{
A constructor function that creates module instances.
}
\description{
Create an nn_module class
}
\examples{
\donttest{
Linear <- nn_module("Linear",
  initialize = function(in_f, out_f) {
    self$register_parameter("weight", torch_randn(c(out_f, in_f)))
  },
  forward = function(x) torch_matmul(x, self$weight$t())
)
m <- Linear(3, 2)
m(torch_randn(c(1, 3)))
}
}
