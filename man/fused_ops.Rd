% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_ops}
\alias{fused_ops}
\title{Execute Fused Operation Chain}
\usage{
fused_ops(tensor, ops)
}
\arguments{
\item{tensor}{Input torch tensor}

\item{ops}{Character vector of operations}
}
\value{
Output torch tensor
}
\description{
Execute Fused Operation Chain
}
\examples{
\donttest{
fused_ops(torch_randn(c(2, 3)), c("relu", "sigmoid"))
}
}
