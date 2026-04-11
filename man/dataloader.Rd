% tinyrox says don't edit this manually, but it can't stop you!
\name{dataloader}
\alias{dataloader}
\title{Create a dataloader}
\usage{
dataloader(dataset, batch_size = 1L, shuffle = FALSE, drop_last = FALSE)
}
\arguments{
\item{dataset}{A torch_dataset object.}

\item{batch_size}{Batch size. Default 1.}

\item{shuffle}{Whether to shuffle indices each epoch. Default FALSE.}

\item{drop_last}{Drop the last incomplete batch. Default FALSE.}
}
\value{
A dataloader object.
}
\description{
Create a dataloader
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(10, 3)))
dl <- dataloader(ds, batch_size = 2L)
length(dl)
}
}
}
