% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_dropout}
\alias{nn_dropout}
\title{Dropout module (no-op in eval mode, tinytorch has no autograd)}
\usage{
nn_dropout(p = 0.5, inplace = FALSE)
}
\arguments{
\item{p}{Probability of dropping. Ignored.}

\item{inplace}{Ignored.}
}
\value{
An nn_module instance.
}
\description{
Dropout module (no-op in eval mode, tinytorch has no autograd)
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_dropout(p = 0.2)
input <- torch_randn(20, 16)
output <- m(input)
}
}
}
