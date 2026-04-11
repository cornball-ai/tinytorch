% tinyrox says don't edit this manually, but it can't stop you!
\name{enumerate}
\alias{enumerate}
\title{Iterate with index}
\usage{
enumerate(x)
}
\arguments{
\item{x}{An iterable (list, vector, or iterator).}
}
\value{
A list of (index, value) pairs.
}
\description{
Iterate with index
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(4, 2)))
dl <- dataloader(ds, batch_size = 2L)
for (b in enumerate(dl)) { }
}
}
}
