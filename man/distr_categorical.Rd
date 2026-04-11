% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_categorical}
\alias{distr_categorical}
\title{Categorical distribution}
\usage{
distr_categorical(probs = NULL, logits = NULL)
}
\arguments{
\item{probs}{Category probabilities (tensor).}

\item{logits}{Unnormalized log probabilities (tensor).}
}
\value{
A distribution object.
}
\description{
Categorical distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_categorical(torch_tensor(c(0.25, 0.25, 0.25, 0.25)))
m$sample() # equal probability of 1,2,3,4
}
}
}
