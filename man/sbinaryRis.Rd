% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sbinaryRis.R
\name{sbinaryRis}
\alias{sbinaryRis}
\alias{terms.binris}
\alias{estfun.binris}
\alias{bread.binris}
\alias{df.residual.binris}
\alias{vcov.binris}
\alias{coef.binris}
\alias{logLik.binris}
\alias{print.binris}
\alias{summary.binris}
\alias{print.summary.binris}
\title{Estimation of spatial probit model for binary outcomes using RIS (GHK) simulator}
\usage{
sbinaryRis(
  formula,
  data,
  subset,
  na.action,
  listw = NULL,
  R = 20,
  model = c("SAR", "SEM"),
  varcov = c("invsigma", "sigma"),
  approximation = TRUE,
  pw = 5,
  start = NULL,
  Qneg = FALSE,
  print.init = FALSE,
  ...
)

\method{terms}{binris}(x, ...)

\method{estfun}{binris}(x, ...)

\method{bread}{binris}(x, ...)

\method{df.residual}{binris}(object, ...)

\method{vcov}{binris}(object, ...)

\method{coef}{binris}(object, ...)

\method{logLik}{binris}(object, ...)

\method{print}{binris}(x, ...)

\method{summary}{binris}(object, eigentol = 1e-12, ...)

\method{print}{summary.binris}(x, digits = max(3, getOption("digits") - 2), ...)
}
\arguments{
\item{formula}{A symbolic description of the model of the form \code{y ~ x | wx} where \code{y} is the binary dependent variable, \code{x} are the independent variables. The variables after \code{|} are those variables that enter spatially lagged: \eqn{WX}.}

\item{data}{A \code{data.frame} containing the variables in the model.}

\item{subset}{An optional vector specifying a subset of observations to be used in the fitting process.}

\item{na.action}{A function which indicates what should happen when the data contains \code{NA}s.}

\item{listw}{Object. An object of class \code{listw}, \code{matrix}, or \code{Matrix}.}

\item{R}{Integer. The number of draws used in RIS (GHK) simulator.}

\item{model}{String. A string indicating which model to estimate. It can be \code{"SAR"} for the spatial autoregressive spatial model or \code{"SEM"} for the spatial error model.}

\item{varcov}{String. A string indicating over which variance-covariance matrix to apply the Chokesly factorization.}

\item{approximation}{Logical. If \code{TRUE} (the default) then \eqn{(I - \lambda W)^{-1}} or \eqn{(I - \rho W)^{-1}} is approximated as \eqn{I + \lambda W + \lambda^2 W^2 + \lambda^3 W^3 + ... +\lambda^q W^q}. If  \code{FALSE}, then the inverse is computed without approximations.}

\item{pw}{Integer. The power used for the approximation \eqn{I + \lambda W + \lambda^2 W^2 + \lambda^3 W^3 + ... +\lambda^q W^q}. The default is 5.}

\item{start}{If not \code{NULL}, the user must provide a vector of initial parameters for the optimization procedure. When \code{start = NULL}, \code{sbinaryRis} uses the traditional Probit estimates as initial values for the parameters, and the correlation between \eqn{y} and \eqn{Wy} as initial value for \eqn{\lambda} or \eqn{\rho}.}

\item{Qneg}{Logical. Whether to construct the negative of the diagonal elements of \eqn{Q}. If \code{Qneg = FALSE}, then \eqn{q_{ii} = 2y_{i} - 1}. If  \code{Qneg = TRUE}, then \eqn{q_{ii} = 1- 2y_{i}}.}

\item{print.init}{Logical. If \code{TRUE} the initial parameters used in the optimization of the first step are printed.}

\item{...}{additional arguments passed to \code{maxLik}.}

\item{x, object, }{An object of class \code{bingmm}.}

\item{eigentol}{The standard errors are only calculated if the ratio of the smallest and largest eigenvalue of the Hessian matrix is less than \code{eigentol}.  Otherwise the Hessian is treated as singular.}

\item{digits}{The number of digits for \code{summary} methods.}
}
\value{
An object of class "\code{binris}", a list with elements:
\item{varcov}{the matrix over which the Chokesly factorization is applied,}
\item{model}{type of model that was fitted,}
\item{Qneg}{matrix Q used,}
\item{approximation}{a logical value indicating whether approximation was used to compute the inverse matrix,}
\item{pw}{the powers for the approximation,}
\item{call}{the matched call,}
\item{X}{the X matrix, which contains also WX if the second part of the \code{formula} is used, }
\item{y}{the dependent variable,}
\item{listw}{the spatial weight matrix,}
\item{formula}{the formula,}
\item{R}{number of draws,}
\item{mf}{model frame,}
\item{data}{the data, }
\item{contrastsX}{the contrasts used in the first part of the formula,}
\item{contrastsD}{the contrasts used in the second part of the formula,}
\item{Xlevels}{a record of the levels of the factors used in fitting.}
}
\description{
Estimation of spatial probit model  using RIS-normal (a.k.a GHK) simulator. The models can be the SAR or SEM probit model model. The SAR probit model has the following structure: 

\deqn{
y^*= X\beta + WX\gamma + \lambda W y^* + \epsilon = Z\delta + \lambda Wy^{*} + \epsilon = A_{\lambda}^{-1}Z\delta + u, 
}
where  \eqn{y = 1} if \eqn{y^*>0} and 0 otherwise, \eqn{Z = (X, WX)}, \eqn{\delta = (\beta', \gamma')'}, \eqn{u = A_{\lambda}^{-1}\epsilon} with \eqn{A_{\lambda} = (I - \lambda W)}, 
and \eqn{\epsilon \sim N(0, I)}. The SEM probit model has the following structure:

\deqn{
y^* = X\beta + WX\gamma + u = Z\delta  + u
}
where \eqn{y = 1} if \eqn{y^*>0} and 0 otherwise, \eqn{Z = (X, WX)}, \eqn{\delta = (\beta', \gamma')'}, 
\eqn{u = \rho W u + \epsilon} such that \eqn{u = A_{\rho}^{-1}\epsilon}, and \eqn{\epsilon \sim N(0, I)},
}
\details{
The models are estimated by simulating the probabilities using the RIS-normal (GHK) simulator. The aim is to evaluate 
the multivariate density function \eqn{P(\upsilon = Q u < s)}, where \eqn{Q} is a diagonal matrix with entries \eqn{q_{ii} = 2y_i - 1}, and the \eqn{n\times 1} vector
\eqn{s} depends on whether the model is SAR or SEM. If \code{model = "SAR"}, then \eqn{s = QA_{\lambda}^{-1}Z\delta} where \eqn{A_{\lambda} = (I - \lambda W)}; 
if \code{model = "SEM"}, then \eqn{s = QZ\delta}. 

Let \eqn{\Sigma_{\upsilon} = QVar(u)Q'} be the variance-covariance model of the transformed model. If \code{model = "SAR"}
 \eqn{\Sigma_{\upsilon} = Q\Sigma_{\lambda}Q'}, where \eqn{\Sigma_{\lambda} = (A_{\lambda}'A_{\lambda})^{-1}}. 
 If \code{model = "SEM"}, then \eqn{\Sigma_{\upsilon} = Q\Sigma_{\rho}Q'}, where \eqn{\Sigma_{\rho} = (A_{\rho}'A_{\rho})^{-1}}.
 
 Since \eqn{\Sigma_{\upsilon}} is positive definite, there exists a Cholesky decomposition such that
 \eqn{C'C = \Sigma_{\upsilon}^{-1}}, where \eqn{C} is the upper triangular Cholesky matrix and \eqn{\Sigma_{\upsilon}^{-1}} is the precision matrix. 
 Let \eqn{B = C^{-1}}. Then the random vector \eqn{Qu} can be replaced by \eqn{Qu = C^{-1}\eta = B\eta}, where \eqn{\eta} is
 a vector of standard normal variables. Then, the upper limit of the integration becomes \eqn{Qu = B \eta < s}, which 
 can be written as \eqn{\eta < B^{-1}s = \nu}. 
 
 The RIS simulator is implemented by drawing a large number \eqn{R} of random vector \eqn{\eta} and computing \eqn{\eta_{rj}} recursively
 for \eqn{j = 1, \ldots, n}. The parameters are estimated using the simulated maximum likelihood (SML) function:
 
 \deqn{\ln \tilde{L}(\theta) = \ln \left(\frac{1}{R}\sum_{r = 1}^R \tilde{p}_r\right), }
  
 where:
 
 \deqn{\tilde{p}_r = \prod_{j = 1}^n \Phi(\hat{\eta}_{jr}),}
 
 and \eqn{\Phi(\cdot)} is the univariate CDF of the standard normal density. 
 
 By default, \code{sbinaryRis} compute the SML using the Cholesky transformation on \eqn{\Sigma_{\upsilon}^{-1}}, \code{varcov = "invsigma"}. 
 The transformation can also be applied to \eqn{\Sigma_{\upsilon}} using \code{varcov = "sigma"}, which is slower than the previous option. 
 
 This estimator can take several minutes for large datasets. Thus, by default the inverse matrices \eqn{A_{\lambda}^{-1}} and \eqn{A_{\rho}^{-1}}
  are approximated using the Leontief expansion.
}
\examples{
\donttest{
data(oldcol, package = "spdep")

# Create dependent (dummy) variable
COL.OLD$CRIMED <- as.numeric(COL.OLD$CRIME > 35)

# Estimate SAR probit model using RIS simulator using Sigma_v^{-1}
ris_sar <- sbinaryRis(CRIMED ~ INC + HOVAL, 
                     data = COL.OLD,
                     R = 50, 
                     listw = spdep::nb2listw(COL.nb, style = "W"), 
                     print.level = 2)
summary(ris_sar)

# Estimate SAR probit model using RIS simulator using Sigma_v
ris_sar_2  <- sbinaryRis(CRIMED ~ INC + HOVAL, 
                        data = COL.OLD,
                        R = 50, 
                        listw = spdep::nb2listw(COL.nb, style = "W"), 
                        varcov = "sigma", 
                        print.level = 2)
summary(ris_sar_2)

# Estimate SDM probit model using RIS simulator
ris_sdm <- sbinaryRis(CRIMED ~ INC + HOVAL | INC + HOVAL, 
                     data = COL.OLD,
                     R = 50, 
                     listw = spdep::nb2listw(COL.nb, style = "W"), 
                     print.level = 2)
summary(ris_sdm)

# Estimate SEM probit model using RIS simulator
ris_sem <- sbinaryRis(CRIMED ~ INC + HOVAL | INC, 
                     data = COL.OLD,
                     R = 50, 
                     listw = spdep::nb2listw(COL.nb, style = "W"), 
                     model = "SEM")
summary(ris_sem)

}
}
\references{
Beron, K. J., & Vijverberg, W. P. (2004). Probit in a spatial context: a Monte Carlo analysis. In Advances in spatial econometrics: methodology, tools and applications (pp. 169-195). Berlin, Heidelberg: Springer Berlin Heidelberg.

Fleming, M. M. (2004). Techniques for estimating spatially dependent discrete choice models. In Advances in spatial econometrics (pp. 145-168). Springer, Berlin, Heidelberg.

Pace, R. K., & LeSage, J. P. (2016). Fast simulated maximum likelihood estimation of the spatial probit model capable of handling large samples. In Spatial Econometrics: Qualitative and Limited Dependent Variables (pp. 3-34). Emerald Group Publishing Limited.

Piras, G., & Sarrias, M. (2023). One or Two-Step? Evaluating GMM Efficiency for Spatial Binary Probit Models. Journal of choice modelling, 48, 100432.
}
\seealso{
\code{\link[spldv]{sbinaryGMM}}, \code{\link[spldv]{impacts.binris}}.
}
\author{
Mauricio Sarrias and Gianfranco Piras.
}
\keyword{models}
