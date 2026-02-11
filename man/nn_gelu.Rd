% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_gelu}
\alias{nn_gelu}
\title{GELU activation module}
\usage{
nn_gelu()
}
\value{
An nn_module instance.
}
\description{
GELU activation module
}
\examples{
\donttest{
m <- nn_gelu()
m(torch_randn(c(2, 3)))
}
}
