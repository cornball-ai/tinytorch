% tinyrox says don't edit this manually, but it can't stop you!
\name{load}
\alias{load}
\title{Load a Compiled Artifact}
\usage{
load(path)
}
\arguments{
\item{path}{Path to a \code{.torchlang} directory.}
}
\value{
A \code{compiled_module} object.
}
\description{
Loads a previously saved \code{.torchlang} artifact from disk,
rebuilds kernels from the IR, and returns a callable
\code{compiled_module}.
}
\details{
This function masks \code{base::load()}. Use
  \code{base::load()} explicitly if you need to load \code{.RData}
  files.
}
