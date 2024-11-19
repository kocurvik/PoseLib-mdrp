//
// Created by kocur on 19-Nov-24.
//

#include "shared_focal_reldepth_relpose.h"

namespace poselib {
void shared_focal_reldepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                   const std::vector<Eigen::Vector2d> &sigma,
                                   std::vector<ImagePair> *models) {
    size_t rowIdx = 0;
    Eigen::MatrixXd coefficients(9, 10);

    for (size_t i = 0; i < 3; ++i)
    {
        double u1 = x1[i](0), v1 = x1[i](1), u2 = x2[i](0), v2 = x2[i](1);
        double s = sigma[i](1) / sigma[i](0);

        coefficients.row(rowIdx++) << -u1, -v1, -1, 0, 0, 0, 0, 0, 0, s * u2;
        coefficients.row(rowIdx++) << 0, 0, 0, -u1, -v1, -1, 0, 0, 0, s * v2;
        coefficients.row(rowIdx++) << 0, 0, 0, 0, 0, 0, -u1, -v1, -1, s;
    }

    Eigen::Matrix<double, 9, 1> b;
    b = -coefficients.col(9);

    Eigen::Matrix<double, 9, 9> A;
    A = coefficients.leftCols(9);

    Eigen::Matrix<double, 9, 1> g;
    g = A.lu().solve(b);

    if (g.hasNaN())
        return;

    const double g1 = g(0);
    const double g2 = g(1);
    const double g3 = g(2);
    const double g4 = g(3);
    const double g5 = g(4);
    const double g6 = g(5);
    const double g7 = g(6);
    const double g8 = g(7);
    const double g9 = g(8);
    const double g12 = g1 * g1;
    const double g22 = g2 * g2;
    const double g32 = g3 * g3;
    const double g42 = g4 * g4;
    const double g52 = g5 * g5;
    const double g62 = g6 * g6;
    const double g72 = g7 * g7;
    const double g82 = g8 * g8;
    const double g92 = g9 * g9;

    const double coe0 = -g12 * g62 + 2 * g1 * g3 * g4 * g6 - g22 * g62 + 2 * g2 * g3 * g5 * g6 - g32 * g42 - g32 * g52 + g32 + g62;
    const double coe1 = g12 * g52 * g92 - g12 * g52 - 2 * g12 * g5 * g6 * g8 * g9 + g12 * g62 * g82 - g12 * g92 + g12 - 2 * g1 * g2 * g4 * g5 * g92 + 2 * g1 * g2 * g4 * g5 + 2 * g1 * g2 * g4 * g6 * g8 * g9 + 2 * g1 * g2 * g5 * g6 * g7 * g9 - 2 * g1 * g2 * g62 * g7 * g8 + 2 * g1 * g3 * g4 * g5 * g8 * g9 - 2 * g1 * g3 * g4 * g6 * g82 - 2 * g1 * g3 * g52 * g7 * g9 + 2 * g1 * g3 * g5 * g6 * g7 * g8 + 2 * g1 * g3 * g7 * g9 + g22 * g42 * g92 - g22 * g42 - 2 * g22 * g4 * g6 * g7 * g9 + g22 * g62 * g72 - g22 * g92 + g22 - 2 * g2 * g3 * g42 * g8 * g9 + 2 * g2 * g3 * g4 * g5 * g7 * g9 + 2 * g2 * g3 * g4 * g6 * g7 * g8 - 2 * g2 * g3 * g5 * g6 * g72 + 2 * g2 * g3 * g8 * g9 + g32 * g42 * g82 - 2 * g32 * g4 * g5 * g7 * g8 + g32 * g52 * g72 - g32 * g72 - g32 * g82 - g42 * g92 + g42 + 2 * g4 * g6 * g7 * g9 - g52 * g92 + g52 + 2 * g5 * g6 * g8 * g9 - g62 * g72 - g62 * g82 + g92 - 1;
    const double coe2 = -g12 * g82 + 2 * g1 * g2 * g7 * g8 - g22 * g72 - g42 * g82 + 2 * g4 * g5 * g7 * g8 - g52 * g72 + g72 + g82;

    const double b24ac = coe1 * coe1 - 4.0 * coe0 * coe2;

    if (b24ac < 0) {
        return;
    }

    if (b24ac > 0) {
        for (int i = 0; i < 2; ++i) {

            double rootof = (-coe1 + std::pow(-1, i) * std::sqrt(b24ac)) / (2.0 * coe2);

            if (rootof > 0) {
                double focal = std::sqrt(rootof);
                Eigen::MatrixXd K(3, 3);
                K << focal, 0, 0,
                    0, focal, 0,
                    0, 0, 1;

                Eigen::MatrixXd G(3, 3);
                G << g(0), g(1), g(2),
                    g(3), g(4), g(5),
                    g(6), g(7), g(8);

                Eigen::MatrixXd H(3, 3);
                H = K.inverse() * G * K;

                Eigen::JacobiSVD<Eigen::MatrixXd> svd2(H);

                Eigen::Vector3d S = svd2.singularValues();


                Eigen::Matrix3d H2 = H / S(1);
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(H2.transpose()*H2, Eigen::ComputeFullU | Eigen::ComputeFullV);

                Eigen::Vector3d S2 = svd.singularValues();
                Eigen::Matrix3d Vt2 = svd.matrixV();
                Eigen::Matrix3d V2 = Vt2;


                if (V2.determinant() < 0) {
                    V2 *= -1;
                }

                double s1 = S2(0);
                double s3 = S2(2);

                Eigen::Vector3d v1 = V2.col(0);
                Eigen::Vector3d v2 = V2.col(1);
                Eigen::Vector3d v3 = V2.col(2);

                Eigen::Vector3d u1 = (sqrt(1.0 - s3) * v1 + sqrt(s1 - 1.0) * v3) / sqrt(s1 - s3);
                Eigen::Vector3d u2 = (sqrt(1.0 - s3) * v1 - sqrt(s1 - 1.0) * v3) / sqrt(s1 - s3);

                Eigen::Matrix3d U1 = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d W1 = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d U2 = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d W2 = Eigen::Matrix3d::Zero();

                U1.col(0) = v2;
                U1.col(1) = u1;
                U1.col(2) = v2.cross(u1);

                W1.col(0) = H2 * v2;
                W1.col(1) = H2 * u1;
                W1.col(2) = (H2 * v2).cross(H2 * u1);

                U2.col(0) = v2;
                U2.col(1) = u2;
                U2.col(2) = v2.cross(u2);

                W2.col(0) = H2 * v2;
                W2.col(1) = H2 * u2;
                W2.col(2) = (H2 * v2).cross(H2 * u2);

                // # compute the rotation matrices
                Eigen::Matrix3d R1 = W1 * U1.transpose();
                Eigen::Matrix3d R2 = W2 * U2.transpose();

                Eigen::Vector3d n1 = v2.cross(u1);
                Eigen::Vector3d t1 = -(H2 - R1) * n1;

                CameraPose pose1 = CameraPose(R1, t1);
                Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
                models->emplace_back(pose1, camera1, camera1);

                Eigen::Vector3d n2 = v2.cross(u2);
                Eigen::Vector3d t2 = -(H2 - R2) * n2;

                CameraPose pose2 = CameraPose(R2, t2);
                Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
                models->emplace_back(pose2, camera2, camera2);
            }
        }
    }
    return;
}
}

