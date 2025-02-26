//
// Created by kocur on 19-Nov-24.
//

#include "shared_focal_reldepth_relpose.h"

namespace poselib {
void shared_focal_reldepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                   const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models,
                                   const RansacOptions &opt) {
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

                // if rot is more than 5 deg
//                if ((R1.trace() - 1) > 1.99238939618) {
//                    if (focal > opt.min_focal_1 and focal < opt.max_focal_1) {
//                        CameraPose pose1 = CameraPose(R1, t1);
//                        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
//                        models->emplace_back(pose1, camera1, camera1);
//                    }
//                } else {
                CameraPose pose1 = CameraPose(R1, t1);
                Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
                models->emplace_back(pose1, camera1, camera1);

                Eigen::Vector3d n2 = v2.cross(u2);
                Eigen::Vector3d t2 = -(H2 - R2) * n2;

//                if ((R2.trace() - 1) > 1.99238939618) {
//                    if (focal > opt.min_focal_1 and focal < opt.max_focal_1) {
//                        CameraPose pose2 = CameraPose(R2, t2);
//                        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
//                        models->emplace_back(pose2, camera2, camera2);
//                    }
//                } else {
                CameraPose pose2 = CameraPose(R2, t2);
                Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);
                models->emplace_back(pose2, camera2, camera2);
            }
        }
    }
    return;
}

Eigen::MatrixXd solver_p3p_s00f(Eigen::VectorXd d)
{
    Eigen::VectorXd coeffs(18);
    coeffs[0] = d[0];
    coeffs[1] = d[2];
    coeffs[2] = d[1];
    coeffs[3] = d[3];
    coeffs[4] = d[4];
    coeffs[5] = d[5];
    coeffs[6] = d[8];
    coeffs[7] = d[6];
    coeffs[8] = d[9];
    coeffs[9] = d[10];
    coeffs[10] = d[7];
    coeffs[11] = d[11];
    coeffs[12] = d[12];
    coeffs[13] = d[13];
    coeffs[14] = d[15];
    coeffs[15] = d[16];
    coeffs[16] = d[14];
    coeffs[17] = d[17];

    static const int coeffs_ind[] = {0,4,12,0,5,12,13,2,4,0,7,13,2,5,12,16,7,2,13,10,0,16,1,6,14,1,8,14,15,1,9,15,3,6,3,8,14,17,9,3,15,11,1,17,11,17,3};

    static const int C_ind[] = {0,5,8,10,14,16,17,18,20,30,32,34,37,38,40,44,47,48,49,50,51,52,54,59,62,64,68,70,71,75,77,79,81,83,91,92,94,98,101,102,103,104,105,106,110,112,114};

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(9,13);
    for (int i = 0; i < 47; i++) {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Eigen::MatrixXd C0 = C.leftCols(9);
    Eigen::MatrixXd C1 = C.rightCols(4);
    Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
    Eigen::MatrixXd RR(7, 4);
    RR << -C12.bottomRows(3), Eigen::MatrixXd::Identity(4, 4);

    static const int AM_ind[] = {0,1,2,5};
    Eigen::MatrixXd AM(4, 4);
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(3).replicate(4, 1)).eval();

    Eigen::MatrixXd sols(3, 4); // s f d
    int m = 0;
    Eigen::MatrixXcd V0(1, 4);
    // sols.row(0) = D.transpose(); // s
    V0 = V.row(0) / (V.row(1)); //d3

    for (int k = 0; k < 4; ++k) {

        if (abs(D(k).imag()) > 0.01 || D(k).real() < 0.1 || D(k).real() > 10.0 || abs(V0(0, k).imag()) > 0.01 || V0(0, k).real() < 0.0)
            continue;

        double s2 = D(k).real(); // s^2
        double f2 = -(d[2] * s2 + d[3]) / (d[0] * s2 + d[1]); // f^2

        if (f2 < 0.0)
            continue;


        sols(0, m) = std::sqrt(s2);  // s
        sols(1, m) = std::sqrt(f2); // f
        sols(2, m) = V0(0, k).real();   // d3
        ++m;
    }

    sols.conservativeResize(3, m);

    return sols;
}


void shared_focal_s00f_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                               const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models) {
    models->clear();
    models->reserve(4);
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

    Eigen::Matrix3d X1;
    X1.col(0) = depth1[0] * x1h[0];
    X1.col(1) = depth1[1] * x1h[1];
    X1.col(2) = depth1[2] * x1h[2];

    Eigen::Matrix3d X2;
    X2.col(0) = depth2[0] * x2h[0];
    X2.col(1) = depth2[1] * x2h[1];
    X2.col(2) = x2h[2];

    double a[17];

    a[0] = X1(0, 0); a[1] = X1(0, 1); a[2] = X1(0, 2);
    a[3] = X1(1, 0); a[4] = X1(1, 1); a[5] = X1(1, 2);
    a[6] = X1(2, 0); a[7] = X1(2, 1); a[8] = X1(2, 2);

    a[9]  = X2(0, 0); a[10] = X2(0, 1); a[11] = X2(0, 2);
    a[12] = X2(1, 0); a[13] = X2(1, 1); a[14] = X2(1, 2);
    a[15] = X2(2, 0); a[16] = X2(2, 1);

    double b[12];
    b[0] = a[0] - a[1]; b[1] = a[3] - a[4]; b[2] = a[6] - a[7];
    b[3] = a[0] - a[2]; b[4] = a[3] - a[5]; b[5] = a[6] - a[8];
    b[6] = a[1] - a[2]; b[7] = a[4] - a[5]; b[8] = a[7] - a[8];
    b[9]  = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];

    double c[18];
    c[0] = -std::pow(b[11], 2);
    c[1] = std::pow(b[2], 2);
    c[2] = -std::pow(b[9], 2) - std::pow(b[10], 2);
    c[3] = std::pow(b[0], 2) + std::pow(b[1], 2);
    
    c[4] = -1.0;
    c[5] = 2 * a[15];
    c[6] = -std::pow(a[15], 2);
    c[7] = std::pow(b[5], 2);
    c[8] = -std::pow(a[11], 2) - std::pow(a[14], 2);
    c[9] = 2 * a[9] * a[11] + 2 * a[12] * a[14];
    c[10] = -std::pow(a[9], 2) - std::pow(a[12], 2);
    c[11] = std::pow(b[3], 2) + std::pow(b[4], 2);
    
    c[12] = 2 * a[16] - 2 * a[15];
    c[13] = std::pow(a[15], 2) - std::pow(a[16], 2);
    c[14] = std::pow(b[8], 2) - std::pow(b[5], 2);
    c[15] = 2 * a[10] * a[11] - 2 * a[9] * a[11] - 2 * a[12] * a[14] + 2 * a[13] * a[14];
    c[16] = std::pow(a[9], 2) - std::pow(a[10], 2) + std::pow(a[12], 2) - std::pow(a[13], 2);
    c[17] = -std::pow(b[3], 2) - std::pow(b[4], 2) + std::pow(b[6], 2) + std::pow(b[7], 2);

    double d[21];

    d[6] = 1 / (a[6] - a[7]);
    d[0] = (-c[3] * c[8]) * d[6];
    d[1] = (-c[3] * c[9]) * d[6];
    d[2] = (c[2] * c[11] - c[3] * c[10]) * d[6];
    d[3] = (-c[3] * c[4] - c[1] * c[8]) * d[6];
    d[4] = (-c[3] * c[5] - c[1] * c[9]) * d[6];
    d[5] = (c[2] * c[7] - c[3] * c[6] + c[0] * c[11] - c[1] * c[10]) * d[6];
    d[7] = (a[6] * a[16] - 2 * a[6] * a[15] + a[7] * a[15] + a[8] * a[15] - a[8] * a[16]) * d[6];
    
    d[8] = 1 / (2 * (a[6] - a[7]) * (a[15] - a[16]));
    d[9] = (-c[3] * c[15]) * d[8];
    d[10] = (c[2] * c[17] - c[3] * c[16]) * d[8];
    d[11] = (-c[3] * c[12] - c[1] * c[15]) * d[8];
    d[12] = (c[2] * c[14] - c[3] * c[13] + c[0] * c[17] - c[1] * c[16]) * d[8];
    
    d[13] = 1 / (a[6] + a[7] - 2 * a[8]);
    d[14] = (a[8] * a[15] - a[7] * a[15] - a[6] * a[16] + a[8] * a[16]) * d[13];
    d[15] = (c[8] * c[17]) * d[13];
    d[16] = (c[9] * c[17] - c[11] * c[15]) * d[13];
    d[17] = (c[10] * c[17] - c[11] * c[16]) * d[13];
    d[18] = (c[4] * c[17] + c[8] * c[14]) * d[13];
    d[19] = (c[5] * c[17] - c[7] * c[15] + c[9] * c[14] - c[11] * c[12]) * d[13];
    d[20] = (c[6] * c[17] - c[7] * c[16] + c[10] * c[14] - c[11] * c[13]) * d[13];

    Eigen::MatrixXd C0(3, 3);
    C0 << d[2], d[5], d[7],
          d[10], d[12], 1.0,
          d[17], d[20], d[14];

    Eigen::MatrixXd C1(3, 4);
    C1 << d[0]-d[9],  d[3]-d[11], d[1]-d[10],  d[4]-d[12],
            0,     0, d[9],     d[11],
            d[15]-d[9], d[18]-d[11], d[16]-d[10], d[19]-d[12];

    Eigen::MatrixXd C2 = -C0.partialPivLu().solve(C1);

    Eigen::MatrixXd AM(4, 4);
    AM << 0, 0, 1.0, 0,
          0, 0, 0, 1.0,
          C2(0,0), C2(0,1), C2(0,2), C2(0,3),
          C2(1,0), C2(1,1), C2(1,2), C2(1,3);

    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(AM, false);
    Eigen::ArrayXcd D = es.eigenvalues();

    for (int k = 0; k < 4; ++k) {
        
        if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0)
            continue;

        double d3 = 1.0 / D(k).real();

        Eigen::MatrixXd A0(2, 2);
        A0 << (d[3]-d[11])*d3*d3 + (d[4]-d[12])*d3 + d[5], d[7],
                d[12] + d[11]*d3, 1.0;

        Eigen::VectorXd A1(2);
        A1 << (d[0]-d[9])*d3*d3 + (d[1]-d[10])*d3 + d[2], d[10] + d[9]*d3;
        Eigen::VectorXd A2 = -A0.partialPivLu().solve(A1);

        if (A2(0) < 0.0)
            continue;

        double s2 = -(c[1]*A2(0) + c[3])/(c[0]*A2(0) + c[2]);
        if (s2 < 0.001)
            continue;

        double s = std::sqrt(s2);
        double f = std::sqrt(A2(0));
        
        Eigen::Matrix3d Kinv;
        Kinv << 1.0 / f, 0, 0, 0, 1.0 / f, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0]) * Kinv * x2h[0] - s * (depth2[1]) * Kinv * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0]) * Kinv * x2h[0] - s * (d3) * Kinv * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * Kinv * x1h[0] - (depth1[1]) * Kinv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * Kinv * x1h[0] - (depth1[2]) * Kinv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * Kinv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * Kinv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        CameraPose pose = CameraPose(rot, trans);
        Camera camera = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera, camera);
    }

}

}

