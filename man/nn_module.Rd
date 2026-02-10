% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_module}
\alias{nn_module}
\title{Create an nn_module class}
\usage{
nn_module(classname = NULL, ...)
}
\arguments{
\item{classname}{Character name for the module class (optional,
defaults to "nn_module").}

\item{...}{Named functions: initialize, forward, and any other methods.}
}
\value{
A constructor function that creates module instances.
}
\description{
Create an nn_module class
}
