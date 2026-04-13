% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_log10}
\alias{torch_log10}
\title{torch_log10}
\usage{
torch_log10(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\value{
A `torch_tensor`.
}
\description{
torch_log10
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_rand(5)
a
torch_log10(a)
}
}
}
