% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_triu_indices}
\alias{torch_triu_indices}
\title{Triu indices}
\usage{
torch_triu_indices(row, col, offset = 0, dtype = torch_long, device = NULL)
}
\arguments{
\item{row}{An integer.}

\item{col}{An integer.}

\item{offset}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::triu_indices}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_triu_indices(3, 3)
a
a = torch_triu_indices(4, 3, -1)
a
a = torch_triu_indices(4, 3, 1)
a

}
}
}
