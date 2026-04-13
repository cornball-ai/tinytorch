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
\value{
A `torch_tensor`.
}
\description{
ELU activation
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_elu(torch_tensor(c(-1, 0, 1)))
}
}
}
