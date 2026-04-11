% tinyrox says don't edit this manually, but it can't stop you!
\name{dataloader_next}
\alias{dataloader_next}
\title{Get next batch from a dataloader iterator}
\usage{
dataloader_next(iter)
}
\arguments{
\item{iter}{A dataloader_iterator.}
}
\value{
A batch (list of tensors), or NULL if exhausted.
}
\description{
Get next batch from a dataloader iterator
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(10, 3)))
dl <- dataloader(ds, batch_size = 2L)
it <- dataloader_make_iter(dl)
batch <- dataloader_next(it)
}
}
}
