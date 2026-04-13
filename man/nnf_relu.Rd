% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_relu}
\alias{nnf_relu}
\title{ReLU activation (functional)}
\usage{
nnf_relu(input)
}
\arguments{
\item{input}{A torch_tensor.}
}
\value{
A `torch_tensor`.
}
\description{
ReLU activation (functional)
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_relu(torch_tensor(c(-1, 0, 1)))
}
}
}
