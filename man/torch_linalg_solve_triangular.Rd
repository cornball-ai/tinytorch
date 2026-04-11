% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_solve_triangular}
\alias{torch_linalg_solve_triangular}
\alias{linalg_solve_triangular}
\title{Linalg solve triangular}
\usage{
torch_linalg_solve_triangular(self, B, upper, left = TRUE, unitriangular = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{B}{A torch_tensor.}

\item{upper}{A logical value.}

\item{left}{A logical value.}

\item{unitriangular}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_solve_triangular}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
