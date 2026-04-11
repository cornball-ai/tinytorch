% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_chi2}
\alias{distr_chi2}
\title{Chi-squared distribution}
\usage{
distr_chi2(df)
}
\arguments{
\item{df}{Degrees of freedom (tensor or numeric).}
}
\value{
A distribution object.
}
\description{
Chi-squared distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_chi2(torch_tensor(1.0))
m$sample() # Chi2 distributed with shape df=1
torch_tensor(0.1046)
}
}
}
