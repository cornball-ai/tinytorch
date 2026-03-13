% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_promote_types}
\alias{torch_promote_types}
\title{Promote types}
\usage{
torch_promote_types(type1, type2)
}
\arguments{
\item{type1}{A torch dtype (e.g., torch_float32).}

\item{type2}{A torch dtype (e.g., torch_float32).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::promote_types}.
}
