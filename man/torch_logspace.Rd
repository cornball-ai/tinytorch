% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logspace}
\alias{torch_logspace}
\title{Logspace}
\usage{
torch_logspace(start, end, steps, base = 10, dtype = NULL, device = NULL)
}
\arguments{
\item{start}{A numeric scalar.}

\item{end}{A numeric scalar.}

\item{steps}{An integer.}

\item{base}{A numeric value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logspace}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_logspace(start=-10, end=10, steps=5)
torch_logspace(start=0.1, end=1.0, steps=5)
torch_logspace(start=0.1, end=1.0, steps=1)
torch_logspace(start=2, end=2, steps=1, base=2)
}
}
}
