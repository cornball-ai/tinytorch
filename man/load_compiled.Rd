% tinyrox says don't edit this manually, but it can't stop you!
\name{load_compiled}
\alias{load_compiled}
\title{Load a Compiled Artifact}
\usage{
load_compiled(path)
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
