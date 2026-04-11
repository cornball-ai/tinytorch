% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_deg2rad}
\alias{torch_deg2rad}
\title{Deg2rad}
\usage{
torch_deg2rad(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::deg2rad}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(rbind(c(180.0, -180.0), c(360.0, -360.0), c(90.0, -90.0)))
torch_deg2rad(a)
}
}
}
