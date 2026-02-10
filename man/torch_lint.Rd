% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lint}
\alias{torch_lint}
\title{Lint Torch Expressions for Graph Breaks}
\usage{
torch_lint(expr, suggest = TRUE)
}
\arguments{
\item{expr}{An R expression to analyze (quoted or unquoted)}

\item{suggest}{Logical, whether to include suggestions. Default TRUE.}
}
\value{
Invisibly returns a list with lint results. Prints a report.
}
\description{
Analyzes an R expression and reports any operations that would cause
graph breaks, preventing full compilation. Provides suggestions for
improving performance.
}
\examples{
\dontrun{
torch_lint(quote({
  z <- x$matmul(y)
  print(z$shape)
  z$relu()
}))
}

}
