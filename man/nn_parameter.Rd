% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_parameter}
\alias{nn_parameter}
\title{Tag a tensor as an nn parameter}
\usage{
nn_parameter(data, requires_grad = TRUE)
}
\arguments{
\item{data}{A torch_tensor.}

\item{requires_grad}{Whether this parameter requires gradients.}
}
\value{
The tensor with additional class tag.
}
\description{
Tag a tensor as an nn parameter
}
\examples{
\donttest{
if (torch_is_installed()) {
p <- nn_parameter(torch_randn(c(3, 4)))
p$requires_grad
}
}
}
