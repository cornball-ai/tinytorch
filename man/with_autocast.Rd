% tinyrox says don't edit this manually, but it can't stop you!
\name{with_autocast}
\alias{with_autocast}
\title{No-op autocast context manager}
\usage{
with_autocast(device_type = "cpu", dtype = NULL, enabled = TRUE, code)
}
\arguments{
\item{device_type}{Ignored.}

\item{dtype}{Ignored.}

\item{enabled}{Ignored.}

\item{code}{Expression to evaluate.}
}
\description{
Since Rtorch is CPU-only, autocast is a no-op.
}
