#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

//[[Rcpp::export]]

arma::vec get_gamma0(
    arma::uvec user,
    arma::uvec item,
    arma::colvec alpha,
    arma::colvec beta,
    arma::mat C,
    arma::mat D
){
  user = user - 1;
  item = item - 1;
  int n = user.n_elem;
  arma::vec output(n);
  for(int i=0; i<n; ++i){
    output(i) = alpha(user(i)) + beta(item(i)) + arma::dot(C.row(user(i)),D.row(item(i)));
  }
  return(output);
}
