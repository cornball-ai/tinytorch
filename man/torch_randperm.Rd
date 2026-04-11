% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randperm}
\alias{torch_randperm}
\title{Randperm}
\usage{
torch_randperm(n, dtype = torch_long, device = NULL)
}
\arguments{
\item{n}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::randperm}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_randperm(4)
}
}
}
