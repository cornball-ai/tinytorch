% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_broadcast_tensors}
\alias{torch_broadcast_tensors}
\title{Broadcast tensors}
\usage{
torch_broadcast_tensors(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::broadcast_tensors}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_arange(0, 3)$view(c(1, 4))
y = torch_arange(0, 2)$view(c(3, 1))
out = torch_broadcast_tensors(list(x, y))
out[[1]]
}
}
}
