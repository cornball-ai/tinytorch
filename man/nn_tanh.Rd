% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_tanh}
\alias{nn_tanh}
\title{Tanh activation module}
\usage{
nn_tanh()
}
\value{
An nn_module instance.
}
\description{
Tanh activation module
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_tanh()
input <- torch_randn(2)
output <- m(input)
}
}
}
