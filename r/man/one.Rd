% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dplyr-funcs-agg.R
\name{one}
\alias{one}
\title{Get one value from each group}
\usage{
one(...)
}
\arguments{
\item{...}{Unquoted column name to pull values from.}
}
\description{
Returns one arbitrary value from the input for each group. The
function is biased towards non-null values: if there is at least one non-null
value for a certain group, that value is returned, and only if all the values
are null for the group will the function return null.
}
\examples{
\dontrun{
mtcars |>
  arrow_table() |>
  group_by(cyl) |>
  summarize(x = one(disp))
}
}
