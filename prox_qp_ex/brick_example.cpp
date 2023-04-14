#include <iostream>
#include <proxsuite/proxqp/sparse/sparse.hpp>  // get the sparse API of ProxQP
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp>  // for c++14
#include <proxsuite/proxqp/sparse/sparse.hpp>

using namespace proxsuite;
using namespace proxsuite::proxqp;
using proxsuite::nullopt;  // c++17 simply use std::nullopt

using T = double;

int main()
{
  isize dim = 2, n_eq = 0, n_in = 1;
  T p = 0.15;             // level of sparsity
  T conditioning = 10.0;  // conditioning level for H

  // ------------------
  int mass = 1.0;
  double dt = 0.01;

  Eigen::MatrixXd v = Eigen::MatrixXd(dim, 1);
  v << 2.0, -3.0;

  Eigen::MatrixXd qk = Eigen::MatrixXd(dim, 1);
  qk << 1.0, 3.0;

  // ------------------
  // define the problem
  double eps_abs = 1e-9;

  Eigen::MatrixXd gravity = Eigen::MatrixXd(dim, 1);
  gravity << 0.0, 9.8;

  Eigen::MatrixXd J = Eigen::MatrixXd(1, dim);
  J << 0.0, 1.0;

  Eigen::MatrixXd mass_matrix = Eigen::MatrixXd(dim, dim);
  mass_matrix << 1.0, 0.0, 0.0, 1.0;

  mass_matrix = mass * mass_matrix;

  // cost H
  Eigen::MatrixXd H = mass_matrix;
  Eigen::SparseMatrix<double> H_spa(n_in, dim);
  H_spa = H.sparseView();

  Eigen::MatrixXd g = Eigen::VectorXd(dim, 1);
  g = mass_matrix * (dt * gravity - v);

  // inequality constraints C
  Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
  C = -J * dt;

  Eigen::SparseMatrix<double> C_spa(n_in, dim);
  C_spa = C.sparseView();

  Eigen::VectorXd l = Eigen::VectorXd(n_in);
  l << -Eigen::Infinity;  // lower bound

  Eigen::VectorXd u = Eigen::VectorXd(n_in);
  u = J * qk;  // upper bound

  std::cout << "H:\n" << H_spa << std::endl;
  std::cout << "g.T:" << g.transpose() << std::endl;
  std::cout << "C:\n" << C_spa << std::endl;
  std::cout << "l.T:" << l.transpose() << std::endl;
  std::cout << "u.T:" << u.transpose() << std::endl;

  // Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim);

  // auto x_sol = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  // auto b = A * x_sol;

  // auto A = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
  // auto C = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_in, n, p);
  // auto x_sol = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  // auto b = A * x_sol;
  // auto l = C * x_sol;
  // auto u = (l.array() + 10).matrix().eval();

  // design a qp2 object using sparsity masks of H, A and C

  proxsuite::proxqp::sparse::QP<T, int> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  qp.settings.verbose = true;

  qp.init(H_spa, g, nullopt, nullopt, C_spa, l, u);
  qp.solve();

  // update H
  // auto H_new = 2 * H;  // keep the same sparsity structure
  // qp.update(H_new, nullopt, nullopt, nullopt, nullopt, nullopt,
  //           nullopt);  // update H with H_new, it will work
  // qp.solve();
  // // generate H2 with another sparsity structure
  // auto H2 = ::proxsuite::proxqp::utils::rand::sparse_positive_definite_rand(n, conditioning, p);
  // qp.update(H2, nullopt, nullopt, nullopt, nullopt, nullopt,
  //           nullopt);  // nothing will happen
  // // if only a vector changes, then the update takes effect
  // auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  // qp.update(nullopt, g, nullopt, nullopt, nullopt, nullopt, nullopt);
  // qp.solve();  // it solves the problem with another vector
  // // to solve the problem with H2 matrix create a new qp object
  // proxsuite::proxqp::sparse::QP<T, isize> qp2(H2.cast<bool>(), A.cast<bool>(), C.cast<bool>());
  // qp2.init(H2, g_new, A, b, C, l, u);
  // qp2.solve();  // it will solve the new problem
  // // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}
