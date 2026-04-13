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
\value{
The result of evaluating `code`.
}
\description{
Evaluates the expression with gradient computation disabled.
}
\examples{
\donttest{
if (torch_is_installed()) {
with_no_grad({
  x <- torch_randn(c(2, 3))
  torch_relu(x)
})
}
}
}
