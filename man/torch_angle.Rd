% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_angle}
\alias{torch_angle}
\title{Angle}
\usage{
torch_angle(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::angle}.
}
\examples{
\donttest{
if (torch_is_installed()) {

# For real tensors, angle returns 0 (positive) or pi (negative)
torch_angle(torch_tensor(c(-1, 2, -3)))


}
}
}
