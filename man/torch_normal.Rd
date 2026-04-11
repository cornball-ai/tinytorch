% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_normal}
\alias{torch_normal}
\title{Normal}
\usage{
torch_normal(mean, std = 1, generator = NULL)
}
\arguments{
\item{mean}{A torch_tensor.}

\item{std}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::normal}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_normal(mean=0, std=torch_arange(1, 0, -0.1) + 1e-6)
torch_normal(mean=0.5, std=torch_arange(1., 6.))
torch_normal(mean=torch_arange(1., 6.))
torch_normal(2, 3, size=c(1, 4))

}
}
}
