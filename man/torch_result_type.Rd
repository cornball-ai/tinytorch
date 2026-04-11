% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_result_type}
\alias{torch_result_type}
\title{Result type}
\usage{
torch_result_type(tensor1, other)
}
\arguments{
\item{tensor1}{A `torch_tensor`.}

\item{other}{A `torch_tensor`.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::result_type}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_result_type(tensor1 = torch_tensor(c(1, 2), dtype=torch_int()), tensor2 = 1)
}
}
}
