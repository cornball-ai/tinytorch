% tinyrox says don't edit this manually, but it can't stop you!
\name{print.module_tree}
\alias{print.module_tree}
\title{Print Module Tree}
\usage{
\method{print}{module_tree}(x, ...)
}
\arguments{
\item{x}{A module_tree from discover_modules()}

\item{...}{Ignored}
}
\value{
Invisible `x`.
}
\description{
Print Module Tree
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_sequential(nn_linear(10, 5), nn_relu())
tree <- discover_modules(m)
print(tree)
}
}
}
