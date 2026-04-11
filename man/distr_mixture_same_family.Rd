% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_mixture_same_family}
\alias{distr_mixture_same_family}
\title{Mixture of same-family distributions}
\usage{
distr_mixture_same_family(mixture_distribution, component_distribution)
}
\arguments{
\item{mixture_distribution}{Categorical distribution for mixture weights.}

\item{component_distribution}{Batch of component distributions.}
}
\value{
A distribution object.
}
\description{
Mixture of same-family distributions
}
\examples{
\donttest{
if (torch_is_installed()) {
# Construct Gaussian Mixture Model in 1D consisting of 5 equally
# weighted normal distributions
mix <- distr_categorical(torch_ones(5))
comp <- distr_normal(torch_randn(5), torch_rand(5))
gmm <- distr_mixture_same_family(mix, comp)
}
}
}
