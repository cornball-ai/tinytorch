% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_gelu}
\alias{nnf_gelu}
\alias{torch_gelu}
\title{GELU activation}
\usage{
nnf_gelu(self, approximate = "none")

torch_gelu(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\description{
GELU activation
}
\examples{
\dontrun{
nnf_gelu(torch_randn(c(2, 3)))
}
}
