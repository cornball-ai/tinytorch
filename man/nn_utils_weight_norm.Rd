% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_utils_weight_norm}
\alias{nn_utils_weight_norm}
\title{Weight norm (stub)}
\usage{
nn_utils_weight_norm(module, name = "weight", dim = 0L)
}
\arguments{
\item{module}{An nn_module.}

\item{name}{Parameter name. Default "weight".}

\item{dim}{Dimension. Default 0.}
}
\value{
The module (unmodified stub).
}
\description{
Weight norm (stub)
}
\examples{
\donttest{
if (torch_is_installed()) {
x = nn_linear(in_features = 20, out_features = 40)
weight_norm = nn_utils_weight_norm$new(name = 'weight', dim = 2)
weight_norm$apply(x)
x$weight_g$size()
x$weight_v$size()
x$weight

# the recompute() method recomputes the weight using g and v. It must be called
# explicitly inside `forward()`.
weight_norm$recompute(x)

}
}
}
