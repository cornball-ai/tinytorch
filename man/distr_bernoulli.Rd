% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_bernoulli}
\alias{distr_bernoulli}
\title{Bernoulli distribution}
\usage{
distr_bernoulli(probs = NULL, logits = NULL)
}
\arguments{
\item{probs}{Probability of 1 (tensor or numeric). One of probs or logits required.}

\item{logits}{Log-odds (tensor or numeric).}
}
\value{
A distribution object.
}
\description{
Bernoulli distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_bernoulli(0.3)
m$sample() # 30\% chance 1; 70\% chance 0
}
}
}
