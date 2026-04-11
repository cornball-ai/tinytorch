% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_generator}
\alias{torch_generator}
\title{Create a random number generator (stub)}
\usage{
torch_generator()
}
\value{
A generator object (currently an environment placeholder).
}
\description{
Create a random number generator (stub)
}
\examples{
\donttest{
if (torch_is_installed()) {

# Via string
generator <- torch_generator()
generator$current_seed()
generator$set_current_seed(1234567L)
generator$current_seed()

}
}
}
