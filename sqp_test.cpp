#include <gtest/gtest.h>
#include <solvers/sqp.hpp>

using namespace sqp;

namespace sqp_test
{

struct SimpleNLP : public NonLinearProblem<double>
{
  using Vector = NonLinearProblem<double>::Vector;
  using Matrix = NonLinearProblem<double>::Matrix;

  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  Eigen::Vector2d SOLUTION = { 1, 1 };

  SimpleNLP()
  {
    num_var = 2;
    num_constr = 3;
  }

  void objective(const Vector& x, Scalar& obj)
  {
    obj = -x.sum();
  }

  void objective_linearized(const Vector& x, Vector& grad, Scalar& obj)
  {
    grad.resize(num_var);

    objective(x, obj);
    grad << -1, -1;
  }

  void constraint(const Vector& x, Vector& c, Vector& l, Vector& u)
  {
    // 1 <= x0^2 + x1^2 <= 2
    // 0 <= x0
    // 0 <= x1
    c << x.squaredNorm(), x;
    l << 1, 0, 0;
    u << 2, infinity, infinity;
  }

  void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c, Vector& l, Vector& u)
  {
    Jc.resize(3, 2);

    constraint(x, c, l, u);
    Jc << 2 * x.transpose(), Matrix::Identity(2, 2);
  }
};

TEST(SQPTestCase, TestSimpleNLP)
{
  SimpleNLP problem;
  SQP<double> solver;
  Eigen::Vector2d x;

  // feasible initial point
  Eigen::Vector2d x0 = { 1.2, 0.1 };
  Eigen::Vector3d y0 = Eigen::VectorXd::Zero(3);

  solver.settings().max_iter = 100;
  solver.settings().second_order_correction = true;

  solver.solve(problem, x0, y0);
  x = solver.primal_solution();

  solver.info().print();
  std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
  std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

  EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
  EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

TEST(SQPTestCase, SimpleNLP_InfeasibleStart)
{
  SimpleNLP problem;
  SQP<double> solver;
  Eigen::Vector2d x;

  // infeasible initial point
  Eigen::Vector2d x0 = { 2, -1 };
  Eigen::Vector3d y0 = { 1, 1, 1 };

  solver.settings().max_iter = 100;
  solver.settings().second_order_correction = true;

  solver.solve(problem, x0, y0);
  x = solver.primal_solution();

  solver.info().print();
  std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
  std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

  EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
  EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

struct SimpleQP : public NonLinearProblem<double>
{
  using Vector = NonLinearProblem<double>::Vector;
  using Matrix = NonLinearProblem<double>::Matrix;

  double dt;
  Eigen::MatrixXd P = Eigen::MatrixXd(2, 2);
  Eigen::Vector2d q = Eigen::VectorXd(2, 1);
  Eigen::MatrixXd Inertia = Eigen::MatrixXd(1, 2);
  Eigen::MatrixXd mass_matrix = Eigen::MatrixXd(2, 2);
  Eigen::MatrixXd gravity = Eigen::MatrixXd(2, 1);

  Eigen::Vector2d SOLUTION = { 0.0, 0.0 };

  SimpleQP()
  {
    dt = 0.01;
    mass_matrix << 1.0, 0.0, 0.0, 1.0;
    gravity << 0.0, 9.8;
    // P << 4, 1, 1, 2;
    P = mass_matrix;
    Inertia << 0.0, 1.0;
    num_var = 2;
    num_constr = 1;
  }

  void objective(const Vector& x, Scalar& obj) final
  {
    q = mass_matrix * (dt * gravity - x);
    obj = 0.5 * x.dot(P * x) + q.dot(x);
  }

  void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) final
  {
    objective(x, obj);
    grad = P * x + q;
  }

  void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) final
  {
    // Gx
    c << Inertia * dt * x;
    l << -Eigen::Infinity;
    // h
    // u << ;
  }

  void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c, Vector& l, Vector& u) final
  {
    constraint(x, c, l, u);
    // This will be just G
    Jc << Inertia * dt;
    // std::cout << "done linearizing constraint" << std::endl;
  }
};

TEST(SQPTestCase, TestSimpleQP)
{
  SimpleQP problem;
  SQP<double> solver;

  // State = velocity == q_dot
  // y0 = lambda == complimentarity
  Eigen::VectorXd x0 = Eigen::Vector2d(1.0, 4.5);  // this is correct
  Eigen::VectorXd y0 = Eigen::VectorXd(1);
  y0 << 9.81;

  solver.settings().second_order_correction = true;
  solver.solve(problem, x0, y0);

  solver.info().print();
  std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
  std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;

  std::cout << "||||||||||||||||||||||| My test" << std::endl;

  EXPECT_TRUE(solver.primal_solution().isApprox(problem.SOLUTION, 1e-2));
  EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

}  // namespace sqp_test
