% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_sigmoid}
\alias{nn_sigmoid}
\title{Sigmoid activation module}
\usage{
nn_sigmoid()
}
\value{
An nn_module instance.
}
\description{
Sigmoid activation module
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_sigmoid()
input <- torch_randn(2)
output <- m(input)
}
}
}
