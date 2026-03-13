% tinyrox says don't edit this manually, but it can't stop you!
\name{print.traced_module}
\alias{print.traced_module}
\title{Print a Traced Module}
\usage{
\method{print}{traced_module}(x, ...)
}
\arguments{
\item{x}{A traced_module}

\item{...}{Ignored}
}
\value{
Invisibly returns x
}
\description{
Print a Traced Module
}
\examples{
\dontrun{
m <- nn_linear(3, 2)
traced <- trace_module(m, input = torch_randn(c(1, 3)))
print(traced)
}
}
