#include "relpose_mono_3pt.h"

#include "PoseLib/misc/sturm.h"
#include "PoseLib/misc/univariate.h"

#include <iostream>
namespace poselib {

Eigen::MatrixXd solver_p3p_mono_3d(const Eigen::VectorXd &data) {
    // Action =  y
    // Quotient ring basis (V) = 1,x,y,z,
    // Available monomials (RR*V) = x*y,y^2,y*z,1,x,y,z,

    const double *d = data.data();
    Eigen::VectorXd coeffs(18);
    coeffs[0] = std::pow(d[6],2) - 2*d[6]*d[7] + std::pow(d[7],2) + std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2);
    coeffs[1] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[3],2) + 2*d[3]*d[4] - std::pow(d[4],2);
    coeffs[2] = 2*std::pow(d[6],2)*d[15] - 2*d[6]*d[7]*d[15] + 2*std::pow(d[9],2)*d[15] - 2*d[9]*d[10]*d[15] - 2*d[6]*d[7]*d[16] + 2*std::pow(d[7],2)*d[16] - 2*d[9]*d[10]*d[16] + 2*std::pow(d[10],2)*d[16];
    coeffs[3] = std::pow(d[6],2)*std::pow(d[15],2) + std::pow(d[9],2)*std::pow(d[15],2) - 2*d[6]*d[7]*d[15]*d[16] - 2*d[9]*d[10]*d[15]*d[16] + std::pow(d[7],2)*std::pow(d[16],2) + std::pow(d[10],2)*std::pow(d[16],2) + std::pow(d[15],2) - 2*d[15]*d[16] + std::pow(d[16],2);
    coeffs[4] = -2*std::pow(d[0],2)*d[12] + 2*d[0]*d[1]*d[12] - 2*std::pow(d[3],2)*d[12] + 2*d[3]*d[4]*d[12] + 2*d[0]*d[1]*d[13] - 2*std::pow(d[1],2)*d[13] + 2*d[3]*d[4]*d[13] - 2*std::pow(d[4],2)*d[13];
    coeffs[5] = -std::pow(d[0],2)*std::pow(d[12],2) - std::pow(d[3],2)*std::pow(d[12],2) + 2*d[0]*d[1]*d[12]*d[13] + 2*d[3]*d[4]*d[12]*d[13] - std::pow(d[1],2)*std::pow(d[13],2) - std::pow(d[4],2)*std::pow(d[13],2) - std::pow(d[12],2) + 2*d[12]*d[13] - std::pow(d[13],2);
    coeffs[6] = std::pow(d[6],2) - 2*d[6]*d[8] + std::pow(d[8],2) + std::pow(d[9],2) - 2*d[9]*d[11] + std::pow(d[11],2);
    coeffs[7] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[3],2) + 2*d[3]*d[5] - std::pow(d[5],2);
    coeffs[8] = 2*std::pow(d[6],2)*d[15] - 2*d[6]*d[8]*d[15] + 2*std::pow(d[9],2)*d[15] - 2*d[9]*d[11]*d[15] - 2*d[6]*d[8]*d[17] + 2*std::pow(d[8],2)*d[17] - 2*d[9]*d[11]*d[17] + 2*std::pow(d[11],2)*d[17];
    coeffs[9] = std::pow(d[6],2)*std::pow(d[15],2) + std::pow(d[9],2)*std::pow(d[15],2) - 2*d[6]*d[8]*d[15]*d[17] - 2*d[9]*d[11]*d[15]*d[17] + std::pow(d[8],2)*std::pow(d[17],2) + std::pow(d[11],2)*std::pow(d[17],2) + std::pow(d[15],2) - 2*d[15]*d[17] + std::pow(d[17],2);
    coeffs[10] = -2*std::pow(d[0],2)*d[12] + 2*d[0]*d[2]*d[12] - 2*std::pow(d[3],2)*d[12] + 2*d[3]*d[5]*d[12] + 2*d[0]*d[2]*d[14] - 2*std::pow(d[2],2)*d[14] + 2*d[3]*d[5]*d[14] - 2*std::pow(d[5],2)*d[14];
    coeffs[11] = -std::pow(d[0],2)*std::pow(d[12],2) - std::pow(d[3],2)*std::pow(d[12],2) + 2*d[0]*d[2]*d[12]*d[14] + 2*d[3]*d[5]*d[12]*d[14] - std::pow(d[2],2)*std::pow(d[14],2) - std::pow(d[5],2)*std::pow(d[14],2) - std::pow(d[12],2) + 2*d[12]*d[14] - std::pow(d[14],2);
    coeffs[12] = std::pow(d[7],2) - 2*d[7]*d[8] + std::pow(d[8],2) + std::pow(d[10],2) - 2*d[10]*d[11] + std::pow(d[11],2);
    coeffs[13] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[14] = 2*std::pow(d[7],2)*d[16] - 2*d[7]*d[8]*d[16] + 2*std::pow(d[10],2)*d[16] - 2*d[10]*d[11]*d[16] - 2*d[7]*d[8]*d[17] + 2*std::pow(d[8],2)*d[17] - 2*d[10]*d[11]*d[17] + 2*std::pow(d[11],2)*d[17];
    coeffs[15] = std::pow(d[7],2)*std::pow(d[16],2) + std::pow(d[10],2)*std::pow(d[16],2) - 2*d[7]*d[8]*d[16]*d[17] - 2*d[10]*d[11]*d[16]*d[17] + std::pow(d[8],2)*std::pow(d[17],2) + std::pow(d[11],2)*std::pow(d[17],2) + std::pow(d[16],2) - 2*d[16]*d[17] + std::pow(d[17],2);
    coeffs[16] = -2*std::pow(d[1],2)*d[13] + 2*d[1]*d[2]*d[13] - 2*std::pow(d[4],2)*d[13] + 2*d[4]*d[5]*d[13] + 2*d[1]*d[2]*d[14] - 2*std::pow(d[2],2)*d[14] + 2*d[4]*d[5]*d[14] - 2*std::pow(d[5],2)*d[14];
    coeffs[17] = -std::pow(d[1],2)*std::pow(d[13],2) - std::pow(d[4],2)*std::pow(d[13],2) + 2*d[1]*d[2]*d[13]*d[14] + 2*d[4]*d[5]*d[13]*d[14] - std::pow(d[2],2)*std::pow(d[14],2) - std::pow(d[5],2)*std::pow(d[14],2) - std::pow(d[13],2) + 2*d[13]*d[14] - std::pow(d[14],2);

    Eigen::MatrixXd C0(3,3);
    C0 << coeffs[0], coeffs[2], coeffs[3],
        coeffs[6], coeffs[8], coeffs[9],
        coeffs[12], coeffs[14], coeffs[15];

    Eigen::MatrixXd C1(3,3);
    C1 << coeffs[1], coeffs[4], coeffs[5],
        coeffs[7], coeffs[10], coeffs[11],
        coeffs[13], coeffs[16], coeffs[17];

    Eigen::MatrixXd C2 = -C0.partialPivLu().solve(C1);

    double k0 = C2(0,0);
    double k1 = C2(0,1);
    double k2 = C2(0,2);
    double k3 = C2(1,0);
    double k4 = C2(1,1);
    double k5 = C2(1,2);
    double k6 = C2(2,0);
    double k7 = C2(2,1);
    double k8 = C2(2,2);

    double c4 = 1.0 / (k3*k3 - k0*k6);
    double c3 = c4 * (2*k3*k4 - k1*k6 - k0*k7);
    double c2 = c4 * (k4*k4 - k0*k8 - k1*k7 - k2*k6 + 2*k3*k5);
    double c1 = c4 * (2*k4*k5 - k2*k7 - k1*k8);
    double c0 = c4 * (k5*k5 - k2*k8);
//    double roots[4];
//    int n_roots = univariate::solve_quartic_real(c3, c2, c1, c0, roots);

    Eigen::Matrix4d CC;

    CC << 0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        c0, c1, c2, c3;

    Eigen::EigenSolver<Eigen::Matrix4d> es(CC, false);
    Eigen::Matrix<std::complex<double>, 4, 1> D = es.eigenvalues();
    int n_roots = 0;
    double roots[4];
    for (int i = 0; i < 4; ++i) {
        if (std::abs(D(i).imag()) > 1e-8)
            continue;
        roots[n_roots++] = D(i).real();
    }

    Eigen::MatrixXd sols(3, n_roots);
    for (int ii = 0; ii < n_roots; ii++) {
        sols(1,ii) = roots[ii];
        sols(0,ii) = k6*roots[ii]*roots[ii] + k7*roots[ii] + k8;
        sols(2,ii) = (k3*roots[ii]*roots[ii] + k4*roots[ii] + k5)/sols(0,ii);
    }
    return sols;
}
int essential_3pt_mono_depth_impl(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                  const std::vector<Eigen::Vector2d> &sigma, std::vector<CameraPose> *rel_pose) {
    std::vector<Eigen::Vector3d> x1h(3);
    std::vector<Eigen::Vector3d> x2h(3);
    for (int i = 0; i < 3; ++i) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
    }

    double depth1[3];
    double depth2[3];
    for (int i = 0; i < 3; ++i) {
        depth1[i] = sigma[i][0];
        depth2[i] = sigma[i][1];
    }

    Eigen::VectorXd datain(18);
    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[0][1], x1h[1][1], x1h[2][1], x2h[0][0], x2h[1][0], x2h[2][0],
        x2h[0][1], x2h[1][1], x2h[2][1], depth1[0], depth1[1], depth1[2], depth2[0], depth2[1], depth2[2];

    Eigen::MatrixXd sols = solver_p3p_mono_3d(datain);

    size_t num_sols = 0;
    for (int k = 0; k < sols.cols(); ++k) {

        double s = std::sqrt(sols(0, k));
        double u = sols(1, k);
        double v = sols(2, k);

        Eigen::Vector3d v1 = s * (depth2[0] + v) * x2h[0] - s * (depth2[1] + v) * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * x2h[0] - s * (depth2[2] + v) * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * x1h[0] - (depth1[1] + u) * x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * x1h[0] - (depth1[2] + u) * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        CameraPose pose;
        Eigen::Quaterniond q_flip(rot);
        pose.q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
        pose.t = s * (depth2[0] + v) * x2h[0] - (depth1[0] + u) * rot * x1h[0];
//        pose.t.normalize();
        pose.shift = u;
        rel_pose->emplace_back(pose);
        num_sols++;
    }

    return num_sols;
}
int essential_3pt_mono_depth(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                             const std::vector<Eigen::Vector2d> &sigma, std::vector<CameraPose> *rel_pose) {
    rel_pose->reserve(rel_pose->size() + 4);
    essential_3pt_mono_depth_impl(x1, x2, sigma, rel_pose);

    return rel_pose->size();
}
} // namespace poselib