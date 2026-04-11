% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_module_dict}
\alias{nn_module_dict}
\title{Module dictionary}
\usage{
nn_module_dict(modules = list())
}
\arguments{
\item{modules}{Named list of nn_module instances.}
}
\value{
An nn_module that stores sub-modules by name.
}
\description{
Module dictionary
}
\examples{
\donttest{
if (torch_is_installed()) {
nn_module <- nn_module(
  initialize = function() {
    self$dict <- nn_module_dict(list(
      l1 = nn_linear(10, 20),
      l2 = nn_linear(20, 10)
    ))
  },
  forward = function(x) {
    x <- self$dict$l1(x)
    self$dict$l2(x)
  }
)
}
}
}
