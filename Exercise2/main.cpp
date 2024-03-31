#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;


VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b)
{
    unsigned int n = A.rows();
    VectorXd x(n);

    x = A.fullPivLu().solve(b);

    return x;
}

VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b)
{
    int n = A.rows();
    VectorXd x(n);

    x = A.fullPivHouseholderQr().solve(b);

    return x;
}

void Errors(const MatrixXd& A, const VectorXd& b, const VectorXd& solution, double& errRelPALU, double& errRelQR)
{
    VectorXd xPALU = SolveSystemPALU(A, b);
    VectorXd xQR = SolveSystemQR(A, b);

    errRelPALU = (xPALU - solution).norm() / solution.norm();
    errRelQR = (xQR - solution).norm() / solution.norm();
}


int main()
{
    Vector2d solution (-1.0e+0, -1.0e+0);
    cout << "Relative errors:" << endl;

    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1 (-5.169911863249772e-01, 1.672384680188350e-01);

    double errRel1PALU;
    double errRel1QR;

    Errors(A1,b1,solution,errRel1PALU,errRel1QR);

    cout << scientific << "1 - " << "PALU: " << errRel1PALU << " QR: " << errRel1QR << endl;


    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2 (-6.394645785530173e-04, 4.259549612877223e-04);

    double errRel2PALU;
    double errRel2QR;

    Errors(A2,b2,solution,errRel2PALU,errRel2QR);

    cout << scientific << "2 - " << "PALU: " << errRel2PALU << " QR: " << errRel2QR << endl;


    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3 (-6.400391328043042e-10, 4.266924591433963e-10);

    double errRel3PALU;
    double errRel3QR;

    Errors(A3,b3,solution,errRel3PALU,errRel3QR);

    cout << scientific << "3 - " << "PALU: " << errRel3PALU << " QR: " << errRel3QR << endl;


  return 0;
}
