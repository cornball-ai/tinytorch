% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_module_list}
\alias{nn_module_list}
\title{List container for nn_modules}
\usage{
nn_module_list(modules = list())
}
\arguments{
\item{modules}{A list of nn_module instances.}
}
\value{
An nn_module containing the sub-modules.
}
\description{
List container for nn_modules
}
\examples{
\donttest{
if (torch_is_installed()) {

my_module <- nn_module(
  initialize = function() {
    self$linears <- nn_module_list(lapply(1:10, function(x) nn_linear(10, 10)))
  },
  forward = function(x) {
    for (i in 1:length(self$linears)) {
      x <- self$linears[[i]](x)
    }
    x
  }
)
}
}
}
