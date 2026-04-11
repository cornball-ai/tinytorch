% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_relu}
\alias{nn_relu}
\title{ReLU activation module}
\usage{
nn_relu()
}
\value{
An nn_module instance.
}
\description{
ReLU activation module
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_relu()
m(torch_randn(c(2, 3)))
}
}
}
