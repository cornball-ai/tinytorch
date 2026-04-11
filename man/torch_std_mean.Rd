% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_std_mean}
\alias{torch_std_mean}
\title{Std mean}
\usage{
torch_std_mean(self, unbiased = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{unbiased}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::std_mean}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_std_mean(a)


a = torch_randn(c(4, 4))
a
torch_std_mean(a, 1)
}
}
}
