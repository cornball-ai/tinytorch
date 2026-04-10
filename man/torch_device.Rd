% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_device}
\alias{torch_device}
\title{Create a torch device specification}
\usage{
torch_device(type = "cpu", index = NULL)
}
\arguments{
\item{type}{Device type string ("cpu" or "cuda").}

\item{index}{Device index (default 0).}
}
\value{
A device string.
}
\description{
Create a torch device specification
}
