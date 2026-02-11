% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_elu}
\alias{nnf_elu}
\title{ELU activation}
\usage{
nnf_elu(self, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{alpha}{Scale for the negative factor. Default 1.0.}
}
\description{
ELU activation
}
\examples{
\donttest{
nnf_elu(torch_tensor(c(-1, 0, 1)))
}
}
