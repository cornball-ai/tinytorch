% tinyrox says don't edit this manually, but it can't stop you!
\name{as_iterator}
\alias{as_iterator}
\title{Create an iterator from a dataloader}
\usage{
as_iterator(x)
}
\arguments{
\item{x}{A dataloader.}
}
\value{
An iterator.
}
\description{
Create an iterator from a dataloader
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(4, 2)))
dl <- dataloader(ds, batch_size = 2L)
it <- as_iterator(dl)
}
}
}
