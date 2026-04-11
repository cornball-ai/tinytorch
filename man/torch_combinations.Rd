% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_combinations}
\alias{torch_combinations}
\title{Combinations}
\usage{
torch_combinations(self, r = 2, with_replacement = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{r}{An integer.}

\item{with_replacement}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::combinations}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = c(1, 2, 3)
tensor_a = torch_tensor(a)
torch_combinations(tensor_a)
torch_combinations(tensor_a, r=3)
torch_combinations(tensor_a, with_replacement=TRUE)
}
}
}
