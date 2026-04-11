% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_normal}
\alias{distr_normal}
\title{Normal distribution}
\usage{
distr_normal(loc, scale)
}
\arguments{
\item{loc}{Mean (tensor or numeric).}

\item{scale}{Standard deviation (tensor or numeric).}
}
\value{
A distribution object with sample(), log_prob(), entropy() methods.
}
\description{
Normal distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_normal(loc = 0, scale = 1)
m$sample() # normally distributed with loc=0 and scale=1
}
}
}
