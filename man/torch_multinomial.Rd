% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_multinomial}
\alias{torch_multinomial}
\title{Draw samples from a multinomial distribution}
\usage{
torch_multinomial(self, num_samples, replacement = FALSE)
}
\arguments{
\item{self}{Probabilities tensor.}

\item{num_samples}{Number of samples per row.}

\item{replacement}{Whether to sample with replacement. Default FALSE.}
}
\value{
A torch_tensor of indices.
}
\description{
Draw samples from a multinomial distribution
}
\examples{
\donttest{
if (torch_is_installed()) {

weights = torch_tensor(c(0, 10, 3, 0), dtype=torch_float()) # create a tensor of weights
torch_multinomial(weights, 2)
torch_multinomial(weights, 4, replacement=TRUE)
}
}
}
