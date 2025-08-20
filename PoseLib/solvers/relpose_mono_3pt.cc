#include "relpose_mono_3pt.h"

#include "PoseLib/misc/sturm.h"
#include "PoseLib/misc/univariate.h"
#include "p3p_common.h"

#include <iostream>
namespace poselib {

// Eigen::MatrixXd solver_p3p_mono_3d(const Eigen::VectorXd &data) {
std::pair<Eigen::MatrixXd, Eigen::VectorXd> solver_p3p_mono_3d(const Eigen::VectorXd &data) {

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

    Eigen::MatrixXd C2 = -C0.fullPivLu().solve(C1);

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

    double roots[4];
    int n_roots = univariate::solve_quartic_real(c3, c2, c1, c0, roots);
    int m = 0;
    Eigen::MatrixXd sols(3, n_roots);
    for (int ii = 0; ii < n_roots; ii++) {
        double ss = k6*roots[ii]*roots[ii] + k7*roots[ii] + k8;
        if (ss < 0.001)
            continue;
        sols(1,ii) = roots[ii];
        sols(0,ii) = std::sqrt(ss);
        sols(2,ii) = (k3*roots[ii]*roots[ii] + k4*roots[ii] + k5)/ss;
        ++m;
    }
    sols.conservativeResize(3, m);
    return {sols, coeffs};
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

    auto [sols, cc] = solver_p3p_mono_3d(datain);
    size_t num_sols = 0;
    for (int k = 0; k < sols.cols(); ++k) {

        double s = sols(0, k);
        double u = sols(1, k);
        double v = sols(2, k);

        if (depth2[0] + v <=0 || depth2[1] + v <=0 || depth2[2] + v <=0 || depth1[0] + u <=0 || depth1[1] + u <=0 || depth1[2] + u <=0)
            continue;

        double s2 = s*s;
        refine_suv(s2, u, v, cc);
        s = std::sqrt(s2);

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
        Eigen::Vector3d t = s * (depth2[0] + v) * x2h[0] - (depth1[0] + u) * rot * x1h[0];

        CameraPose pose = CameraPose(rot, t);
        pose.shift = u;
        pose.scale = s;
        rel_pose->emplace_back(pose);
        num_sols++;
    }

    return num_sols;
}

int essential_3pt_mono_madpose(const std::vector<Eigen::Vector2d> &xa, const std::vector<Eigen::Vector2d> &xb,
                             const std::vector<Eigen::Vector2d> &sigma, std::vector<CameraPose> *rel_pose) {
    rel_pose->clear();
    rel_pose->reserve(4);
    Eigen::Matrix<double, 3, 3> x1;
    Eigen::Matrix<double, 3, 3> x2;
    

    for (int i = 0; i < 3; ++i) {
        x1.row(i) = xa[i].homogeneous().transpose();
        x2.row(i) = xb[i].homogeneous().transpose();
    }

    Eigen::Vector3d d1;
    Eigen::Vector3d d2;
    for (int i = 0; i < 3; ++i) {
        d1(i) = sigma[i][0];
        d2(i) = sigma[i][1];
    }

    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(18);

    coeffs[0] = 2 * x2.row(0).dot(x2.row(1)) - x2.row(0).dot(x2.row(0)) - x2.row(1).dot(x2.row(1));
    coeffs[1] = x1.row(0).dot(x1.row(0)) + x1.row(1).dot(x1.row(1)) - 2 * x1.row(0).dot(x1.row(1));
    coeffs[2] = 2 * (d2[0] + d2[1]) * x2.row(0).dot(x2.row(1)) - 2 * d2[0] * x2.row(0).dot(x2.row(0)) -
                2 * d2[1] * x2.row(1).dot(x2.row(1));
    coeffs[3] = 2 * d2[0] * d2[1] * x2.row(0).dot(x2.row(1)) - d2[0] * d2[0] * x2.row(0).dot(x2.row(0)) -
                d2[1] * d2[1] * x2.row(1).dot(x2.row(1));
    coeffs[4] = 2 * d1[0] * x1.row(0).dot(x1.row(0)) + 2 * d1[1] * x1.row(1).dot(x1.row(1)) -
                2 * (d1[0] + d1[1]) * x1.row(0).dot(x1.row(1));
    coeffs[5] = d1[0] * d1[0] * x1.row(0).dot(x1.row(0)) + d1[1] * d1[1] * x1.row(1).dot(x1.row(1)) -
                2 * d1[0] * d1[1] * x1.row(0).dot(x1.row(1));
    coeffs[6] = 2 * x2.row(0).dot(x2.row(2)) - x2.row(0).dot(x2.row(0)) - x2.row(2).dot(x2.row(2));
    coeffs[7] = x1.row(0).dot(x1.row(0)) + x1.row(2).dot(x1.row(2)) - 2 * x1.row(0).dot(x1.row(2));
    coeffs[8] = 2 * (d2[0] + d2[2]) * x2.row(0).dot(x2.row(2)) - 2 * d2[0] * x2.row(0).dot(x2.row(0)) -
                2 * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[9] = 2 * d2[0] * d2[2] * x2.row(0).dot(x2.row(2)) - d2[0] * d2[0] * x2.row(0).dot(x2.row(0)) -
                d2[2] * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[10] = 2 * d1[0] * x1.row(0).dot(x1.row(0)) + 2 * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * (d1[0] + d1[2]) * x1.row(0).dot(x1.row(2));
    coeffs[11] = d1[0] * d1[0] * x1.row(0).dot(x1.row(0)) + d1[2] * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * d1[0] * d1[2] * x1.row(0).dot(x1.row(2));
    coeffs[12] = 2 * x2.row(1).dot(x2.row(2)) - x2.row(1).dot(x2.row(1)) - x2.row(2).dot(x2.row(2));
    coeffs[13] = x1.row(1).dot(x1.row(1)) + x1.row(2).dot(x1.row(2)) - 2 * x1.row(1).dot(x1.row(2));
    coeffs[14] = 2 * (d2[1] + d2[2]) * x2.row(1).dot(x2.row(2)) - 2 * d2[1] * x2.row(1).dot(x2.row(1)) -
                 2 * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[15] = 2 * d2[1] * d2[2] * x2.row(1).dot(x2.row(2)) - d2[1] * d2[1] * x2.row(1).dot(x2.row(1)) -
                 d2[2] * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[16] = 2 * d1[1] * x1.row(1).dot(x1.row(1)) + 2 * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * (d1[1] + d1[2]) * x1.row(1).dot(x1.row(2));
    coeffs[17] = d1[1] * d1[1] * x1.row(1).dot(x1.row(1)) + d1[2] * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * d1[1] * d1[2] * x1.row(1).dot(x1.row(2));

    const std::vector<int> coeff_ind0 = {0, 6,  12, 1,  7,  13, 2,  8, 0,  6,  12, 14, 6, 0, 12, 1,  7,  13, 3,
                                         9, 2,  8,  14, 15, 4,  10, 7, 1,  16, 13, 8,  2, 6, 12, 0,  14, 9,  3,
                                         8, 14, 2,  15, 3,  9,  15, 4, 10, 16, 7,  13, 1, 5, 11, 10, 4,  17, 16};
    const std::vector<int> coeff_ind1 = {11, 17, 5, 9, 15, 3, 5, 11, 17, 10, 16, 4, 11, 5, 17};
    const std::vector<int> ind0 = {0,   1,   9,   12,  13,  21,  24,  25,  26,  28,  29,  33,  39,  42,  47,
                                   50,  52,  53,  60,  61,  62,  64,  65,  69,  72,  73,  75,  78,  81,  83,
                                   87,  90,  91,  92,  94,  95,  99,  102, 103, 104, 106, 107, 110, 112, 113,
                                   122, 124, 125, 127, 128, 130, 132, 133, 135, 138, 141, 143};
    const std::vector<int> ind1 = {7, 8, 10, 19, 20, 22, 26, 28, 29, 31, 32, 34, 39, 42, 47};
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(12, 12);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(12, 4);

    for (size_t k = 0; k < ind0.size(); k++) {
        int i = ind0[k] % 12;
        int j = ind0[k] / 12;
        C0(i, j) = coeffs[coeff_ind0[k]];
    }

    for (size_t k = 0; k < ind1.size(); k++) {
        int i = ind1[k] % 12;
        int j = ind1[k] / 12;
        C1(i, j) = coeffs[coeff_ind1[k]];
    }

    Eigen::MatrixXd C2 = C0.colPivHouseholderQr().solve(C1);
    Eigen::Matrix4d AM;
    AM << Eigen::RowVector4d(0, 0, 1, 0), -C2.row(9), -C2.row(10), -C2.row(11);

    Eigen::EigenSolver<Eigen::Matrix4d> es(AM);
    Eigen::Vector4cd D = es.eigenvalues();
    Eigen::Matrix4cd V = es.eigenvectors();

    Eigen::MatrixXcd sols = Eigen::MatrixXcd(4, 3);
    sols.col(0) = V.row(1).array() / V.row(0).array();
    sols.col(1) = D;
    sols.col(2) = V.row(3).array() / V.row(0).array();

    std::vector<Eigen::Vector4d> solutions;
    for (int i = 0; i < 4; i++) {
        if (D[i].imag() != 0)
            continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector4d sol;
        sol << 1.0, b1, a2, b2 * a2;
        solutions.push_back(sol);
    }

    Eigen::Vector3d depth_x = d1;
    Eigen::Vector3d depth_y = d2;
    
    for (auto &sol : solutions) {
        Eigen::Vector3d dd1, dd2;

        dd1 = depth_x.array() + sol(1);
        dd2 = depth_y.array() * sol(2) + sol(3);

        if (dd1.minCoeff() <= 0 || dd2.minCoeff() <= 0)
            continue;

        Eigen::MatrixXd X = x1.transpose().array().rowwise() * dd1.transpose().array();
        Eigen::MatrixXd Y = x2.transpose().array().rowwise() * dd2.transpose().array();

        Eigen::Vector3d centroid_X = X.rowwise().mean();
        Eigen::Vector3d centroid_Y = Y.rowwise().mean();

        Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
        Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

        Eigen::Matrix3d S = Y_centered * X_centered.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        if (U.determinant() * V.determinant() < 0) {
            U.col(2) *= -1;
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::Vector3d t = centroid_Y - R * centroid_X;

        CameraPose pose = CameraPose(R, t);
        pose.shift = sol(1);
        rel_pose->emplace_back(pose);
    }

    return rel_pose->size();
}

int essential_3pt_mono_depth(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                             const std::vector<Eigen::Vector2d> &sigma, std::vector<CameraPose> *rel_pose) {
    rel_pose->clear();
    rel_pose->reserve(4);
    essential_3pt_mono_depth_impl(x1, x2, sigma, rel_pose);

    return rel_pose->size();
}
} // namespace poselib
