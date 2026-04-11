% tinyrox says don't edit this manually, but it can't stop you!
\name{find_torch_packages}
\alias{find_torch_packages}
\title{Find Installed Packages That Depend on torch}
\usage{
find_torch_packages()
}
\value{
Character vector of package names
}
\description{
Searches installed R packages for those that import or depend on torch.
}
\examples{
\donttest{
if (torch_is_installed()) {
find_torch_packages()
}
}
}
