% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_kl_div}
\alias{torch_kl_div}
\title{Kl div}
\usage{
torch_kl_div(self, target, reduction = 1L, log_target = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{reduction}{An integer.}

\item{log_target}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::kl_div}.
}
