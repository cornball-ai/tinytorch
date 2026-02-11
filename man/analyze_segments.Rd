% tinyrox says don't edit this manually, but it can't stop you!
\name{analyze_segments}
\alias{analyze_segments}
\title{Analyze Expression Segments}
\usage{
analyze_segments(expr)
}
\arguments{
\item{expr}{An R expression}
}
\value{
A data.frame with segment info
}
\description{
Provides a summary of how an expression would be segmented.
}
\examples{
analyze_segments(quote({ y <- x$relu(); print(y) }))
}
