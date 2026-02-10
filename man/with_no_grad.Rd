% tinyrox says don't edit this manually, but it can't stop you!
\name{with_no_grad}
\alias{with_no_grad}
\title{No-op gradient context manager}
\usage{
with_no_grad(code)
}
\arguments{
\item{code}{Expression to evaluate.}
}
\description{
Since Rtorch has no autograd, this simply evaluates the expression.
}
