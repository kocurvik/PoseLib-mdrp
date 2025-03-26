#ifndef POSELIB_P3P_COMMON_H
#define POSELIB_P3P_COMMON_H

#include <cmath>

namespace poselib {

bool inline root2real(double b, double c, double &r1, double &r2) {
    double THRESHOLD = -1.0e-12;
    double v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    double y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

inline std::array<Eigen::Vector3d, 2> compute_pq(Eigen::Matrix3d C) {
    std::array<Eigen::Vector3d, 2> pq;
    Eigen::Matrix3d C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);

    pq[0] = C.col(0);
    pq[1] = C.row(0);

    return pq;
}

// Performs a few newton steps on the equations
inline void refine_lambda(double &lambda1, double &lambda2, double &lambda3, const double a12, const double a13,
                          const double a23, const double b12, const double b13, const double b23) {

    for (int iter = 0; iter < 5; ++iter) {
        double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
            return;
        double x11 = lambda1 - lambda2 * b12;
        double x12 = lambda2 - lambda1 * b12;
        double x21 = lambda1 - lambda3 * b13;
        double x23 = lambda3 - lambda1 * b13;
        double x32 = lambda2 - lambda3 * b23;
        double x33 = lambda3 - lambda2 * b23;
        double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
        // This uses the closed form of the inverse for the jacobean.
        // Due to the zero elements this actually becomes quite nice.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

// Performs a few newton steps on the equations
inline void refine_suv(double &s, double &u, double &v, const Eigen::VectorXd c) {
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Vector3d r;
        r(0) = c(0) * s * v * v + c(1) * u * u + c(2) * s * v + c(3) * s + c(4) * u + c(5);
        r(1) = c(6) * s * v * v + c(7) * u * u + c(8) * s * v + c(9) * s + c(10) * u + c(11);
        r(2) = c(12) * s * v * v + c(13) * u * u + c(14) * s * v + c(15) * s + c(16) * u + c(17);
        if (std::abs(r(0)) + std::abs(r(1)) + std::abs(r(2)) < 1e-10)
            return;
        Eigen::Matrix3d J;
        J(0, 0) = c(0) * v * v + c(2) * v + c(3);
        J(0, 1) = 2.0 * c(1) * u + c(4);
        J(0, 2) = 2.0 * c(0) * s * v + c(2) * s;

        J(1, 0) = c(6) * v * v + c(8) * v + c(9);
        J(1, 1) = 2.0 * c(7) * u + c(10);
        J(1, 2) = 2.0 * c(6) * s * v + c(8) * s;

        J(2, 0) = c(12) * v * v + c(14) * v + c(15);
        J(2, 1) = 2.0 * c(13) * u + c(16);
        J(2, 2) = 2.0 * c(12) * s * v + c(14) * s;

        Eigen::Vector3d delta_lambda = (J.transpose() * J).ldlt().solve(-J.transpose() * r);

        s += delta_lambda(0);
        u += delta_lambda(1);
        v += delta_lambda(2);
    }
}

inline void estimateRigidTransformSVD(
    const std::vector<Eigen::Vector3d> &P,
    const std::vector<Eigen::Vector3d> &Q,
    Eigen::Matrix3d &R,
    Eigen::Vector3d &t) 
{
    int N = P.size();

    Eigen::Vector3d cP = Eigen::Vector3d::Zero();
    Eigen::Vector3d cQ = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; i++) {
        cP += P[i];
        cQ += Q[i];
    }
    cP /= N;
    cQ /= N;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        H += (P[i] - cP) * (Q[i] - cQ).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    t = cQ - R * cP;
}

} // namespace poselib

#endif // POSELIB_P3P_COMMON_H
