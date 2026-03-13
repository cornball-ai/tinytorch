% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_channel_shuffle}
\alias{torch_native_channel_shuffle}
\title{Native channel shuffle}
\usage{
torch_native_channel_shuffle(self, groups)
}
\arguments{
\item{self}{A torch_tensor.}

\item{groups}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_channel_shuffle}.
}
