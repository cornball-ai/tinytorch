% tinyrox says don't edit this manually, but it can't stop you!
\name{find_modules_in_package}
\alias{find_modules_in_package}
\title{Find nn_module Definitions in an R Package}
\usage{
find_modules_in_package(pkg)
}
\arguments{
\item{pkg}{Package name (string) or path to package source directory}
}
\value{
A data.frame with columns: name, file, exported
}
\description{
Searches a package's R source files for nn_module definitions.
Works on both installed packages and local source directories.
}
\examples{
\dontrun{
find_modules_in_package("tinytorch")
}
}
