% tinyrox says don't edit this manually, but it can't stop you!
\name{dataloader_make_iter}
\alias{dataloader_make_iter}
\title{Create an iterator from a dataloader}
\usage{
dataloader_make_iter(dl)
}
\arguments{
\item{dl}{A dataloader object.}
}
\value{
An iterator environment with .next() method.
}
\description{
Create an iterator from a dataloader
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(10, 3)))
dl <- dataloader(ds, batch_size = 2L)
it <- dataloader_make_iter(dl)
}
}
}
