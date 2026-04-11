% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lerp}
\alias{torch_lerp}
\title{Lerp}
\usage{
torch_lerp(self, end, weight)
}
\arguments{
\item{self}{A torch_tensor.}

\item{end}{A torch_tensor.}

\item{weight}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lerp}.
}
\examples{
\donttest{
if (torch_is_installed()) {

start = torch_arange(1, 4)
end = torch_empty(4)$fill_(10)
start
end
torch_lerp(start, end, 0.5)
torch_lerp(start, end, torch_full_like(start, 0.5))
}
}
}
