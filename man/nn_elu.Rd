% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_elu}
\alias{nn_elu}
\title{ELU activation module}
\usage{
nn_elu(alpha = 1)
}
\arguments{
\item{alpha}{Scale for the negative factor. Default 1.0.}
}
\value{
An nn_module instance.
}
\description{
ELU activation module
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_elu()
input <- torch_randn(2)
output <- m(input)
}
}
}
