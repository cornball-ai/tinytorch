% tinyrox says don't edit this manually, but it can't stop you!
\name{with_no_grad}
\alias{with_no_grad}
\title{Disable gradient computation}
\usage{
with_no_grad(code)
}
\arguments{
\item{code}{Expression to evaluate.}
}
\description{
Evaluates the expression with gradient computation disabled.
}
\examples{
\dontrun{
with_no_grad({
  x <- torch_randn(c(2, 3))
  torch_relu(x)
})
}
}
