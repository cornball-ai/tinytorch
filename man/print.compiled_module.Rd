% tinyrox says don't edit this manually, but it can't stop you!
\name{print.compiled_module}
\alias{print.compiled_module}
\title{Print a Compiled Module}
\usage{
\method{print}{compiled_module}(x, ...)
}
\arguments{
\item{x}{A \code{compiled_module}}

\item{...}{Ignored}
}
\value{
Invisibly returns \code{x}
}
\description{
Print a Compiled Module
}
\examples{
\donttest{
m <- nn_linear(3, 2)
cm <- compile(m, input = torch_randn(c(1, 3)))
print(cm)
}
}
