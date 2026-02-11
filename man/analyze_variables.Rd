% tinyrox says don't edit this manually, but it can't stop you!
\name{analyze_variables}
\alias{analyze_variables}
\title{Compilation Layer for torchlang}
\usage{
analyze_variables(expr)
}
\arguments{
\item{expr}{An R expression}
}
\value{
A list with `reads` and `writes` character vectors
}
\description{
Compiles graph-safe segments using jit_trace, with fallback to
direct execution for R code segments.
Analyze Variable Flow in an Expression
Identifies which variables are read (inputs) and written (outputs)
in an expression. Used to determine function signatures for compilation.
}
\examples{
analyze_variables(quote(y <- x$matmul(w)))
}
