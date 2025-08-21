//
// Created by kocur on 19-Nov-24.
//

#include "varying_focal_monodepth_relpose.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/misc/univariate.h"
#include "relpose_7pt.h"

#include <iostream>
namespace poselib {

void varying_focal_fundamental_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                       std::vector<ImagePair> *models, const RansacOptions &opt) {

    std::vector<Eigen::Vector3d> x1h(7);
    std::vector<Eigen::Vector3d> x2h(7);
    for (int i = 0; i < 7; ++i) {
        x1h[i] = x1[i].homogeneous().normalized();
        x2h[i] = x2[i].homogeneous().normalized();
    }

    std::vector<Eigen::Matrix3d> Fs;
    relpose_7pt(x1h, x2h, &Fs);

    models->reserve(Fs.size());

    for (const Eigen::Matrix3d& F : Fs) {
        std::pair<Camera, Camera> cameras =
            focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
        Camera camera1 = cameras.first;
        Camera camera2 = cameras.second;

        const double focal1 = camera1.focal();
        const double focal2 = camera2.focal();

        if (std::isnan(focal1))
            continue;
        if (std::isnan(focal2))
            continue;

//        if (focal1 < opt.max_focal_1 or focal1 > opt.max_focal_1 or
//            focal2 < opt.min_focal_2 or focal2 > opt.max_focal_2)
//            continue;

        Eigen::DiagonalMatrix<double, 3> K1(focal1, focal1, 1.0);
        Eigen::DiagonalMatrix<double, 3> K2(focal2, focal2, 1.0);

        Eigen::Matrix3d E = K2 * F * K1;

        std::vector<CameraPose> poses;
        motion_from_essential(E, x1h, x2h, &poses);

        for (const CameraPose &pose : poses) {
            models->emplace_back(pose, camera1, camera2);
        }
    }
}

void varying_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                     const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models,
                                     const RansacOptions &opt) {
    models->clear();
    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = x1[i].homogeneous().normalized();
        x2h[i] = x2[i].homogeneous().normalized();
    }

    Eigen::MatrixXd coefficients(12, 12);
    int i;

    // Form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    size_t row = 0;
    for (i = 0; i < 4; i++)
    {
        double u11 = x1[i](0), v11 = x1[i](1), u12 = x2[i](0), v12 = x2[i](1);
        double q1 = sigma[i](0), q2 = sigma[i](1);
        double q = q2 / q1;

        coefficients(row, 0) = -u11;
        coefficients(row, 1) = -v11;
        coefficients(row, 2) = -1;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = 0;
        coefficients(row, 10) = q;
        coefficients(row, 11) = -q * v12;
        ++row;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = -u11;
        coefficients(row, 4) = -v11;
        coefficients(row, 5) = -1;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = -q;
        coefficients(row, 10) = 0;
        coefficients(row, 11) = q * u12;
        ++row;

        if (i == 3)
            break;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = -u11;
        coefficients(row, 7) = -v11;
        coefficients(row, 8) = -1;
        coefficients(row, 9) = q * v12;
        coefficients(row, 10) = -q * u12;
        coefficients(row, 11) = 0;
        ++row;
    }

    Eigen::Matrix<double, 12, 1> f1 = coefficients.block<11, 11>(0, 0).partialPivLu().solve(-coefficients.block<11, 1>(0, 11)).homogeneous();

    Eigen::Matrix3d F;
    F << f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8];

//    std::cout << "F: " << std::endl << F << std::endl;
//    std::cout << "Ep: " << x2h[0].transpose() * F * x1h[0] << std::endl;
//    std::cout << "Det: " << F.determinant() << std::endl;

    std::pair<Camera, Camera> cameras = focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

    Camera camera1 = cameras.first;
    Camera camera2 = cameras.second;

    const double focal1 = camera1.focal();
    const double focal2 = camera2.focal();

    if (std::isnan(focal1))
        return;
    if (std::isnan(focal2))
        return;

//    if (focal1 < opt.max_focal_1 or focal1 > opt.max_focal_1 or
//        focal2 < opt.min_focal_2 or focal2 > opt.max_focal_2)
//        return;

    Eigen::DiagonalMatrix<double, 3> K1(focal1, focal1, 1.0);
    Eigen::DiagonalMatrix<double, 3> K2(focal2, focal2, 1.0);

    Eigen::Matrix3d E = K2 * F * K1;

    std::vector<CameraPose> poses;
    motion_from_essential(E, x1h, x2h, &poses);

    models->reserve(poses.size());

    for (const CameraPose& pose : poses){
        models->emplace_back(pose, camera1, camera2);
    }
}

Eigen::MatrixXd varying_focal_monodepth_gj(Eigen::VectorXd d){
    Eigen::VectorXd coeffs(40);
    coeffs[0] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] - std::pow(d[5], 2);
    coeffs[1] = std::pow(d[8], 2) - 2 * d[8] * d[9] + std::pow(d[9], 2) + std::pow(d[12], 2) - 2 * d[12] * d[13] + std::pow(d[13], 2);
    coeffs[2] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[1] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[5] * d[16] + 2 * d[0] * d[1] * d[17] - 2 * std::pow(d[1], 2) * d[17] + 2 * d[4] * d[5] * d[17] - 2 * std::pow(d[5], 2) * d[17];
    coeffs[3] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[9] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[13] * d[20] - 2 * d[8] * d[9] * d[21] + 2 * std::pow(d[9], 2) * d[21] - 2 * d[12] * d[13] * d[21] + 2 * std::pow(d[13], 2) * d[21];
    coeffs[4] = std::pow(d[20], 2) - 2 * d[20] * d[21] + std::pow(d[21], 2);
    coeffs[5] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[1] * d[16] * d[17] + 2 * d[4] * d[5] * d[16] * d[17] - std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2);
    coeffs[6] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[9] * d[20] * d[21] - 2 * d[12] * d[13] * d[20] * d[21] + std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2);
    coeffs[7] = -std::pow(d[16], 2) + 2 * d[16] * d[17] - std::pow(d[17], 2);
    coeffs[8] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[6] - std::pow(d[6], 2);
    coeffs[9] = std::pow(d[8], 2) - 2 * d[8] * d[10] + std::pow(d[10], 2) + std::pow(d[12], 2) - 2 * d[12] * d[14] + std::pow(d[14], 2);
    coeffs[10] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[2] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[6] * d[16] + 2 * d[0] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[4] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
    coeffs[11] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[10] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[14] * d[20] - 2 * d[8] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[12] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
    coeffs[12] = std::pow(d[20], 2) - 2 * d[20] * d[22] + std::pow(d[22], 2);
    coeffs[13] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[2] * d[16] * d[18] + 2 * d[4] * d[6] * d[16] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
    coeffs[14] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[10] * d[20] * d[22] - 2 * d[12] * d[14] * d[20] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
    coeffs[15] = -std::pow(d[16], 2) + 2 * d[16] * d[18] - std::pow(d[18], 2);
    coeffs[16] = -std::pow(d[0], 2) + 2 * d[0] * d[3] - std::pow(d[3], 2) - std::pow(d[4], 2) + 2 * d[4] * d[7] - std::pow(d[7], 2);
    coeffs[17] = std::pow(d[8], 2) - 2 * d[8] * d[11] + std::pow(d[11], 2) + std::pow(d[12], 2) - 2 * d[12] * d[15] + std::pow(d[15], 2);
    coeffs[18] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[3] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[7] * d[16] + 2 * d[0] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[4] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
    coeffs[19] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[11] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[15] * d[20] - 2 * d[8] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[12] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
    coeffs[20] = std::pow(d[20], 2) - 2 * d[20] * d[23] + std::pow(d[23], 2);
    coeffs[21] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[3] * d[16] * d[19] + 2 * d[4] * d[7] * d[16] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
    coeffs[22] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[11] * d[20] * d[23] - 2 * d[12] * d[15] * d[20] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
    coeffs[23] = -std::pow(d[16], 2) + 2 * d[16] * d[19] - std::pow(d[19], 2);
    coeffs[24] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[5], 2) + 2 * d[5] * d[6] - std::pow(d[6], 2);
    coeffs[25] = std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2) + std::pow(d[13], 2) - 2 * d[13] * d[14] + std::pow(d[14], 2);
    coeffs[26] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[2] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[6] * d[17] + 2 * d[1] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[5] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
    coeffs[27] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[10] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[14] * d[21] - 2 * d[9] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[13] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
    coeffs[28] = std::pow(d[21], 2) - 2 * d[21] * d[22] + std::pow(d[22], 2);
    coeffs[29] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[2] * d[17] * d[18] + 2 * d[5] * d[6] * d[17] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
    coeffs[30] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[10] * d[21] * d[22] - 2 * d[13] * d[14] * d[21] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
    coeffs[31] = -std::pow(d[17], 2) + 2 * d[17] * d[18] - std::pow(d[18], 2);
    coeffs[32] = -std::pow(d[1], 2) + 2 * d[1] * d[3] - std::pow(d[3], 2) - std::pow(d[5], 2) + 2 * d[5] * d[7] - std::pow(d[7], 2);
    coeffs[33] = std::pow(d[9], 2) - 2 * d[9] * d[11] + std::pow(d[11], 2) + std::pow(d[13], 2) - 2 * d[13] * d[15] + std::pow(d[15], 2);
    coeffs[34] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[3] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[7] * d[17] + 2 * d[1] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[5] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
    coeffs[35] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[11] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[15] * d[21] - 2 * d[9] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[13] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
    coeffs[36] = std::pow(d[21], 2) - 2 * d[21] * d[23] + std::pow(d[23], 2);
    coeffs[37] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[3] * d[17] * d[19] + 2 * d[5] * d[7] * d[17] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
    coeffs[38] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[11] * d[21] * d[23] - 2 * d[13] * d[15] * d[21] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
    coeffs[39] = -std::pow(d[17], 2) + 2 * d[17] * d[19] - std::pow(d[19], 2);


    static const int coeffs_ind[] = {0, 8, 16, 24, 32, 1, 9, 17, 25, 33, 0, 8, 16, 24, 32, 2, 10, 8, 0, 18, 16, 24, 26, 32, 34, 3, 11, 1, 9, 19, 17, 25, 27, 33, 35, 9, 1, 17, 25, 33, 4, 12, 20, 28, 36, 2, 10, 0, 18, 16,
												 26, 8, 24, 34, 32, 5, 13, 10, 2, 21, 18, 26, 29, 34, 37, 6, 14, 3, 11, 22, 19, 27, 30, 35, 38, 11, 3, 1, 19, 17, 27, 9, 25, 35, 33, 4, 12, 20, 28, 36, 12, 4, 20, 28, 36, 13, 5, 21, 29, 37,
												 14, 6, 3, 22, 19, 30, 11, 27, 38, 35, 4, 20, 12, 28, 36, 7, 15, 23, 31, 39, 7, 15, 23, 31, 39, 5, 13, 2, 21, 18, 29, 10, 26, 37, 34, 6, 14, 22, 30, 38, 7, 23, 15, 31, 39, 15, 7, 23, 31, 39,
												 5, 21, 13, 29, 37, 6, 22, 14, 30, 38};

    static const int C_ind[] = {0, 1, 6, 13, 19, 20, 21, 26, 33, 39, 42, 44, 49, 51, 57, 60, 61, 63, 65, 66, 68, 72, 73, 76, 79, 80, 81, 82, 84, 86, 89, 91, 93, 97, 99, 103, 105, 108, 112, 116, 120, 121, 126, 133, 139, 142, 144, 147, 149, 150,
                                151, 154, 155, 157, 158, 160, 161, 163, 165, 166, 168, 172, 173, 176, 179, 180, 181, 182, 184, 186, 189, 191, 193, 197, 199, 203, 205, 207, 208, 210, 212, 214, 215, 216, 218, 222, 224, 229, 231, 237, 243, 245, 248, 252, 256, 263, 265, 268, 272, 276,
                                283, 285, 287, 288, 290, 292, 294, 295, 296, 298, 307, 310, 314, 315, 318, 322, 324, 329, 331, 337, 340, 341, 346, 353, 359, 362, 364, 367, 369, 370, 371, 374, 375, 377, 378, 382, 384, 389, 391, 397, 407, 410, 414, 415, 418, 423, 425, 428, 432, 436,
                                447, 450, 454, 455, 458, 467, 470, 474, 475, 478};

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(20, 24);
    for (int i = 0; i < 160; i++)
    {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Eigen::MatrixXd C0 = C.leftCols(20);
    Eigen::MatrixXd C1 = C.rightCols(4);
    Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
    Eigen::MatrixXd RR(8, 4);
    RR << -C12.bottomRows(4), Eigen::MatrixXd::Identity(4, 4);

    static const int AM_ind[] = {0, 1, 2, 3};
    Eigen::MatrixXd AM(4, 4);
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).replicate(4, 1)).eval();

    Eigen::MatrixXd sols(5, 4);
    int m = 0;
    for (int k = 0; k < 4; ++k)
    {

        if (abs(D(k).imag()) > 0.001 || V(2, k).real() < 0.0 ||
            V(3, k).real() < 0.0 || abs(V(2, k).imag()) > 0.001 || abs(V(3, k).imag()) > 0.001)
            continue;

        sols(1, m) = D(k).real();	 // u
        sols(2, m) = V(1, k).real(); // v
        sols(3, m) = V(2, k).real(); // f
        sols(4, m) = V(3, k).real(); // m

        double ss = -(coeffs[0] * sols(1, m) * sols(1, m) * sols(3, m) + coeffs[1] * sols(2, m) * sols(2, m) * sols(4, m) + coeffs[2] * sols(1, m) * sols(3, m) + coeffs[3] * sols(2, m) * sols(4, m) + coeffs[5] * sols(3, m) + coeffs[6] * sols(4, m) + coeffs[7]) / coeffs[4];

        if (ss < 0)
            continue;
        sols(0, m) = std::sqrt(ss); // s
        sols(3, m) = 1.0 / std::sqrt(V(2, k).real());
        sols(4, m) = 1.0 / std::sqrt(V(3, k).real() / ss);
        ++m;
    }

    sols.conservativeResize(5,m);
    return sols;
}


Eigen::MatrixXd varying_focal_monodepth_eigen(Eigen::VectorXd d){
    Eigen::VectorXd coeffs(48);
    coeffs[0] = std::pow(d[8],2) - 2*d[8]*d[9] + std::pow(d[9],2) + std::pow(d[12],2) - 2*d[12]*d[13] + std::pow(d[13],2);
    coeffs[1] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[2] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[9]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[13]*d[20] - 2*d[8]*d[9]*d[21] + 2*std::pow(d[9],2)*d[21] - 2*d[12]*d[13]*d[21] + 2*std::pow(d[13],2)*d[21];
    coeffs[3] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[1]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[5]*d[16] + 2*d[0]*d[1]*d[17] - 2*std::pow(d[1],2)*d[17] + 2*d[4]*d[5]*d[17] - 2*std::pow(d[5],2)*d[17];
    coeffs[4] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[9]*d[20]*d[21] - 2*d[12]*d[13]*d[20]*d[21] + std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2);
    coeffs[5] = std::pow(d[20],2) - 2*d[20]*d[21] + std::pow(d[21],2);
    coeffs[6] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[1]*d[16]*d[17] + 2*d[4]*d[5]*d[16]*d[17] - std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2);
    coeffs[7] = -std::pow(d[16],2) + 2*d[16]*d[17] - std::pow(d[17],2);
    coeffs[8] = std::pow(d[8],2) - 2*d[8]*d[10] + std::pow(d[10],2) + std::pow(d[12],2) - 2*d[12]*d[14] + std::pow(d[14],2);
    coeffs[9] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[6] - std::pow(d[6],2);
    coeffs[10] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[10]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[14]*d[20] - 2*d[8]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[12]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[11] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[2]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[6]*d[16] + 2*d[0]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[4]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[12] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[10]*d[20]*d[22] - 2*d[12]*d[14]*d[20]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[13] = std::pow(d[20],2) - 2*d[20]*d[22] + std::pow(d[22],2);
    coeffs[14] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[2]*d[16]*d[18] + 2*d[4]*d[6]*d[16]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[15] = -std::pow(d[16],2) + 2*d[16]*d[18] - std::pow(d[18],2);
    coeffs[16] = std::pow(d[8],2) - 2*d[8]*d[11] + std::pow(d[11],2) + std::pow(d[12],2) - 2*d[12]*d[15] + std::pow(d[15],2);
    coeffs[17] = -std::pow(d[0],2) + 2*d[0]*d[3] - std::pow(d[3],2) - std::pow(d[4],2) + 2*d[4]*d[7] - std::pow(d[7],2);
    coeffs[18] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[11]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[15]*d[20] - 2*d[8]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[12]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[19] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[3]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[7]*d[16] + 2*d[0]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[4]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[20] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[11]*d[20]*d[23] - 2*d[12]*d[15]*d[20]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[21] = std::pow(d[20],2) - 2*d[20]*d[23] + std::pow(d[23],2);
    coeffs[22] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[3]*d[16]*d[19] + 2*d[4]*d[7]*d[16]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[23] = -std::pow(d[16],2) + 2*d[16]*d[19] - std::pow(d[19],2);
    coeffs[24] = std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2) + std::pow(d[13],2) - 2*d[13]*d[14] + std::pow(d[14],2);
    coeffs[25] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[5],2) + 2*d[5]*d[6] - std::pow(d[6],2);
    coeffs[26] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[10]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[14]*d[21] - 2*d[9]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[13]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[27] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[2]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[6]*d[17] + 2*d[1]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[5]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[28] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[10]*d[21]*d[22] - 2*d[13]*d[14]*d[21]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[29] = std::pow(d[21],2) - 2*d[21]*d[22] + std::pow(d[22],2);
    coeffs[30] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[2]*d[17]*d[18] + 2*d[5]*d[6]*d[17]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[31] = -std::pow(d[17],2) + 2*d[17]*d[18] - std::pow(d[18],2);
    coeffs[32] = std::pow(d[9],2) - 2*d[9]*d[11] + std::pow(d[11],2) + std::pow(d[13],2) - 2*d[13]*d[15] + std::pow(d[15],2);
    coeffs[33] = -std::pow(d[1],2) + 2*d[1]*d[3] - std::pow(d[3],2) - std::pow(d[5],2) + 2*d[5]*d[7] - std::pow(d[7],2);
    coeffs[34] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[11]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[15]*d[21] - 2*d[9]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[13]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[35] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[3]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[7]*d[17] + 2*d[1]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[5]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[36] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[11]*d[21]*d[23] - 2*d[13]*d[15]*d[21]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[37] = std::pow(d[21],2) - 2*d[21]*d[23] + std::pow(d[23],2);
    coeffs[38] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[3]*d[17]*d[19] + 2*d[5]*d[7]*d[17]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[39] = -std::pow(d[17],2) + 2*d[17]*d[19] - std::pow(d[19],2);
    coeffs[40] = std::pow(d[10],2) - 2*d[10]*d[11] + std::pow(d[11],2) + std::pow(d[14],2) - 2*d[14]*d[15] + std::pow(d[15],2);
    coeffs[41] = -std::pow(d[2],2) + 2*d[2]*d[3] - std::pow(d[3],2) - std::pow(d[6],2) + 2*d[6]*d[7] - std::pow(d[7],2);
    coeffs[42] = 2*std::pow(d[10],2)*d[22] - 2*d[10]*d[11]*d[22] + 2*std::pow(d[14],2)*d[22] - 2*d[14]*d[15]*d[22] - 2*d[10]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[14]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[43] = -2*std::pow(d[2],2)*d[18] + 2*d[2]*d[3]*d[18] - 2*std::pow(d[6],2)*d[18] + 2*d[6]*d[7]*d[18] + 2*d[2]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[6]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[44] = std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2) - 2*d[10]*d[11]*d[22]*d[23] - 2*d[14]*d[15]*d[22]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[45] = std::pow(d[22],2) - 2*d[22]*d[23] + std::pow(d[23],2);
    coeffs[46] = -std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2) + 2*d[2]*d[3]*d[18]*d[19] + 2*d[6]*d[7]*d[18]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[47] = -std::pow(d[18],2) + 2*d[18]*d[19] - std::pow(d[19],2);


    Eigen::MatrixXd C0(6, 6);
    C0 << coeffs[1], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7],
        coeffs[9], coeffs[11], coeffs[12], coeffs[13], coeffs[14], coeffs[15],
        coeffs[17], coeffs[19], coeffs[20], coeffs[21], coeffs[22], coeffs[23],
        coeffs[25], coeffs[27], coeffs[28], coeffs[29], coeffs[30], coeffs[31],
        coeffs[33], coeffs[35], coeffs[36], coeffs[37], coeffs[38], coeffs[39],
        coeffs[41], coeffs[43], coeffs[44], coeffs[45], coeffs[46], coeffs[47];


    Eigen::MatrixXd C1(6, 2);
    C1 << coeffs[0], coeffs[2],
        coeffs[8], coeffs[10],
        coeffs[16], coeffs[18],
        coeffs[24], coeffs[26],
        coeffs[32], coeffs[34],
        coeffs[40], coeffs[42];

    Eigen::MatrixXd C12 = -C0.partialPivLu().solve(C1);
    Eigen::MatrixXd AM(2, 2);
    AM << 0.0, 1.0,
        C12(2,0), C12(2,1);

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();

    Eigen::MatrixXd sols(5, 2);
    int m = 0;
    for (int k = 0; k < 2; ++k)
    {

        if (abs(D(k).imag()) > 0.001)
            continue;

        sols(2, m) = 1.0 / D(k).real(); // v x3
        Eigen::MatrixXd A1(5, 5);
        double v2 = sols(2, m)*sols(2, m);
        A1 << coeffs[1], coeffs[3], coeffs[0]*v2+coeffs[2]*sols(2, m)+coeffs[4], coeffs[5], coeffs[6],
            coeffs[9], coeffs[11], coeffs[8]*v2+coeffs[10]*sols(2, m)+coeffs[12], coeffs[13], coeffs[14],
            coeffs[17], coeffs[19], coeffs[16]*v2+coeffs[18]*sols(2, m)+coeffs[20], coeffs[21], coeffs[22],
            coeffs[25], coeffs[27], coeffs[24]*v2+coeffs[26]*sols(2, m)+coeffs[28], coeffs[29], coeffs[30],
            coeffs[33], coeffs[35], coeffs[32]*v2+coeffs[34]*sols(2, m)+coeffs[36], coeffs[37], coeffs[38];
        Eigen::VectorXd A0(5, 1);
        A0 << -coeffs[7],
            -coeffs[15],
            -coeffs[23],
            -coeffs[31],
            -coeffs[39];
        Eigen::VectorXd xz = A1.partialPivLu().solve(A0);

        if (xz[3] < 0.0 || xz[4] < 0.0 || xz[2]/xz[3] < 0.0)
            continue;

        sols(0, m) = std::sqrt(xz[3]); // s x1
        sols(3, m) = 1.0 / std::sqrt(xz[4]); // f x4
        sols(1, m) = xz[1]/xz[4]; // u x2
        sols(4, m) = 1.0 / std::sqrt(xz[2]/xz[3]); // m x5
        ++m;
    }

    sols.conservativeResize(5,m);
    return sols;
}


void varying_focal_monodepth_abspose_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma,
                                          std::vector<ImagePair> *models, const RansacOptions &opt){
    models->clear();
    models->reserve(3);
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
    double a[17];
    a[0] = x1h[0][0]*depth1[0];
    a[1] = x1h[1][0]*depth1[1];
    a[2] = x1h[2][0]*depth1[2];
    a[3] = x1h[0][1]*depth1[0];
    a[4] = x1h[1][1]*depth1[1];
    a[5] = x1h[2][1]*depth1[2];
    a[6] = depth1[0];
    a[7] = depth1[1];
    a[8] = depth1[2];

    a[9] = x2h[0][0]*depth2[0];
    a[10] = x2h[1][0]*depth2[1];
    a[11] = x2h[2][0];
    a[12] = x2h[0][1]*depth2[0];
    a[13] = x2h[1][1]*depth2[1];
    a[14] = x2h[2][1];
    a[15] = depth2[0];
    a[16] = depth2[1];

    double b[12];
    b[0] = a[0] - a[1];
    b[1] = a[3] - a[4];
    b[2] = a[6] - a[7];
    b[3] = a[0] - a[2];
    b[4] = a[3] - a[5];
    b[5] = a[6] - a[8];
    b[6] = a[1] - a[2];
    b[7] = a[4] - a[5];
    b[8] = a[7] - a[8];
    b[9] = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];

    double c[19];

    c[0] = 1.0 / (b[0] * b[0] + b[1] * b[1]);
    c[1] = (-b[9] * b[9] - b[10] * b[10]) * c[0];
    c[2] = (b[2] * b[2] - b[11] * b[11]) * c[0];
    c[3] = 1.0 / (b[3] * b[3] + b[4] * b[4]);
    c[4] = (-a[11] * a[11] - a[14] * a[14]) * c[3];
    c[5] = 2.0 * (a[9] * a[11] + a[12] * a[14]) * c[3];
    c[6] = (-a[9] * a[9] - a[12] * a[12]) * c[3];
    c[7] = (-1.0) * c[3];
    c[8] = (2.0 * a[15]) * c[3];
    c[9] = (b[5] * b[5] - a[15] * a[15]) * c[3];
    c[10] = 1.0 / (-b[3] * b[3] - b[4] * b[4] + b[6] * b[6] + b[7] * b[7]);
    c[11] = 2.0 * (a[10] * a[11] - a[9] * a[11] - a[12] * a[14] + a[13] * a[14]) * c[10];
    c[12] = (a[9] * a[9] - a[10] * a[10] + a[12] * a[12] - a[13] * a[13]) * c[10];
    c[13] = 2.0 * (a[16] - a[15]) * c[10];
    c[14] = (a[15] * a[15] - a[16] * a[16] - b[5] * b[5] + b[8] * b[8]) * c[10];
    c[15] = c[6] - c[1];
    c[16] = c[9] - c[2];
    c[17] = c[12] - c[1];
    c[18] = c[14] - c[2];

    double d[4];
    d[3] = 1.0 / (c[4] * c[13] - c[7] * c[11]);
    d[2] = d[3] * (c[5] * c[13] - c[8] * c[11] + c[4] * c[18] - c[7] * c[17]);
    d[1] = d[3] * (c[5] * c[18] - c[8] * c[17] - c[11] * c[16] + c[13] * c[15]);
    d[0] = d[3] * (c[15] * c[18] - c[16] * c[17]);

    double roots[3];
    int num = univariate::solve_cubic_real(d[2], d[1], d[0], roots);

    for (int k = 0; k < num; ++k){
        double d3 = roots[k];
        if (d3 < 0)
            continue;
        double w = -(c[7]*d3*d3 + c[8]*d3 + c[16]) / (c[4]*d3*d3 + c[5]*d3 + c[15]);
        if (w < 0)
            continue;

        // [1, c1, c2] [f^2, m^2, 1]
        double f = -(c[1]*w + c[2]);
        if (f < 0)
            continue;
        w = std::sqrt(w);
        f = std::sqrt(f);

        double focal1 = 1.0 / f;
        double focal2 = 1.0 / w;

//        if (focal1 < opt.max_focal_1 or focal1 > opt.max_focal_1 or
//            focal2 < opt.min_focal_2 or focal2 > opt.max_focal_2)
//            continue;

        Eigen::Matrix3d K1inv;
        K1inv << f, 0, 0,
            0, f, 0,
            0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << w, 0, 0,
            0,  w, 0,
            0, 0, 1;

        Eigen::Vector3d v1 = (depth2[0]) * K2inv*x2h[0] - (depth2[1]) * K2inv*x2h[1];
        Eigen::Vector3d v2 = (depth2[0]) * K2inv*x2h[0] - (d3) * K2inv*x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * K1inv*x1h[0] - (depth1[1]) * K1inv*x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * K1inv*x1h[0] - (depth1[2]) * K1inv*x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * K1inv*x1h[0];
        Eigen::Vector3d trans2 = (depth2[0]) * K2inv*x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        // if (focal1 > 4.0 || focal1 < 0.2)
        // focal1 = 1.2;
        // if (focal2 > 4.0 || focal2 < 0.2)
        // focal2 = 1.2;

        CameraPose pose = CameraPose(rot, trans);
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal1, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }
}

void varying_focal_monodepth_relpose_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                          std::vector<ImagePair> *models){
    models->clear();
    models->reserve(4);
    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
    }

    double depth1[4];
    double depth2[4];
    for (int i = 0; i < 4; ++i) {
        depth1[i] = sigma[i][0];
        depth2[i] = sigma[i][1];
    }

    Eigen::VectorXd datain(24);
    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[3][0], x1h[0][1], x1h[1][1], x1h[2][1], x1h[3][1], x2h[0][0],
        x2h[1][0], x2h[2][0], x2h[3][0], x2h[0][1], x2h[1][1], x2h[2][1], x2h[3][1], depth1[0], depth1[1], depth1[2],
        depth1[3], depth2[0], depth2[1], depth2[2], depth2[3];

    Eigen::MatrixXd sols;
    if (use_eigen)
        sols = varying_focal_monodepth_eigen(datain);
    else
        sols = varying_focal_monodepth_gj(datain);

    models->reserve(sols.cols());

    for (int k = 0; k < sols.cols(); ++k){
        double s = sols(0, k);
        double u = sols(1, k);
        double v = sols(2, k);
        double f = sols(3, k);
        double w = sols(4, k);

        Eigen::Matrix3d K1inv;
        K1inv << 1.0 / f, 0, 0,
            0, 1.0 / f, 0,
            0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << 1.0 / w, 0, 0,
            0, 1.0 / w, 0,
            0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0] + v) * K2inv*x2h[0] - s * (depth2[1] + v) * K2inv*x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * K2inv*x2h[0] - s * (depth2[2] + v) * K2inv*x2h[2];
        if (depth2[0] + v <= 0 || depth2[1] + v <= 0 || depth2[2] + v <= 0)
            continue;
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * K1inv*x1h[0] - (depth1[1] + u) * K1inv*x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * K1inv*x1h[0] - (depth1[2] + u) * K1inv*x1h[2];
        if (depth1[0] + u <= 0 || depth1[1] + u <= 0 || depth1[2] + u <= 0)
            continue;
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0] + u) * rot * K1inv*x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0] + v) * K2inv*x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        CameraPose pose = CameraPose(rot, trans);
        pose.shift_1 = u;
        pose.shift_2 = v;
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{w, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }
}

void varying_focal_monodepth_relpose_madpose(const std::vector<Eigen::Vector2d> &xa, const std::vector<Eigen::Vector2d> &xb,
                                    const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models) {
    models->clear();
    models->reserve(4);

    Eigen::Matrix<double, 4, 3> x1;
    Eigen::Matrix<double, 4, 3> x2;


    for (int i = 0; i < 4; ++i) {
        x1.row(i) = xa[i].homogeneous().transpose();
        x2.row(i) = xb[i].homogeneous().transpose();
    }

    Eigen::Vector4d d1;
    Eigen::Vector4d d2;
    for (int i = 0; i < 4; ++i) {
        d1(i) = sigma[i][0];
        d2(i) = sigma[i][1];
    }

    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(40);

    coeffs[0] = 2 * x2(0, 0) * x2(1, 0) + 2 * x2(0, 1) * x2(1, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1);
    coeffs[1] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(1, 1) - 2 * x1(0, 0) * x1(1, 0) + x1(0, 1) * x1(0, 1) +
                x1(1, 0) * x1(1, 0) + x1(1, 1) * x1(1, 1);
    coeffs[2] = 2 * d2(0) * x2(0, 0) * x2(1, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) -
                2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(1) * x2(0, 0) * x2(1, 0) +
                2 * d2(0) * x2(0, 1) * x2(1, 1) + 2 * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[3] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(1) * x1(1, 0) * x1(1, 0) +
                2 * d1(1) * x1(1, 1) * x1(1, 1) - 2 * d1(0) * x1(0, 0) * x1(1, 0) - 2 * d1(1) * x1(0, 0) * x1(1, 0) -
                2 * d1(0) * x1(0, 1) * x1(1, 1) - 2 * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[4] = 2 * d2(0) * d2(1) * x2(0, 0) * x2(1, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                d2(1) * d2(1) * x2(1, 0) * x2(1, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[5] = 2 * d2(0) * d2(1) - d2(0) * d2(0) - d2(1) * d2(1);
    coeffs[6] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) -
                2 * d1(0) * d1(1) * x1(0, 0) * x1(1, 0) - 2 * d1(0) * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[7] = d1(0) * d1(0) - 2 * d1(0) * d1(1) + d1(1) * d1(1);
    coeffs[8] = 2 * x2(0, 0) * x2(2, 0) + 2 * x2(0, 1) * x2(2, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[9] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(2, 1) - 2 * x1(0, 0) * x1(2, 0) + x1(0, 1) * x1(0, 1) +
                x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[10] = 2 * d2(0) * x2(0, 0) * x2(2, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(2, 1) +
                 2 * d2(2) * x2(0, 0) * x2(2, 0) + 2 * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[11] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(0) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * x1(0, 1) * x1(2, 1) -
                 2 * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[12] = 2 * d2(0) * d2(2) * x2(0, 0) * x2(2, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[13] = 2 * d2(0) * d2(2) - d2(0) * d2(0) - d2(2) * d2(2);
    coeffs[14] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(0) * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[15] = d1(0) * d1(0) - 2 * d1(0) * d1(2) + d1(2) * d1(2);
    coeffs[16] = 2 * x2(1, 0) * x2(2, 0) + 2 * x2(1, 1) * x2(2, 1) - x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1) -
                 x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[17] = x1(1, 0) * x1(1, 0) - 2 * x1(1, 1) * x1(2, 1) - 2 * x1(1, 0) * x1(2, 0) + x1(1, 1) * x1(1, 1) +
                 x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[18] = 2 * d2(1) * x2(1, 0) * x2(2, 0) - 2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(2) * x2(1, 0) * x2(2, 0) +
                 2 * d2(1) * x2(1, 1) * x2(2, 1) + 2 * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[19] = 2 * d1(1) * x1(1, 0) * x1(1, 0) + 2 * d1(1) * x1(1, 1) * x1(1, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(1) * x1(1, 0) * x1(2, 0) - 2 * d1(2) * x1(1, 0) * x1(2, 0) -
                 2 * d1(1) * x1(1, 1) * x1(2, 1) - 2 * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[20] = 2 * d2(1) * d2(2) * x2(1, 0) * x2(2, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(1) * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[21] = 2 * d2(1) * d2(2) - d2(1) * d2(1) - d2(2) * d2(2);
    coeffs[22] = d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(1) * d1(2) * x1(1, 0) * x1(2, 0) - 2 * d1(1) * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[23] = d1(1) * d1(1) - 2 * d1(1) * d1(2) + d1(2) * d1(2);
    coeffs[24] = 2 * x2(0, 0) * x2(3, 0) + 2 * x2(0, 1) * x2(3, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                 x2(3, 0) * x2(3, 0) - x2(3, 1) * x2(3, 1);
    coeffs[25] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(3, 1) - 2 * x1(0, 0) * x1(3, 0) + x1(0, 1) * x1(0, 1) +
                 x1(3, 0) * x1(3, 0) + x1(3, 1) * x1(3, 1);
    coeffs[26] = 2 * d2(0) * x2(0, 0) * x2(3, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(3) * x2(3, 0) * x2(3, 0) -
                 2 * d2(3) * x2(3, 1) * x2(3, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(3, 1) +
                 2 * d2(3) * x2(0, 0) * x2(3, 0) + 2 * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[27] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(3) * x1(3, 0) * x1(3, 0) +
                 2 * d1(3) * x1(3, 1) * x1(3, 1) - 2 * d1(0) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * x1(0, 1) * x1(3, 1) -
                 2 * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[28] = 2 * d2(0) * d2(3) * x2(0, 0) * x2(3, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(3) * d2(3) * x2(3, 0) * x2(3, 0) - d2(3) * d2(3) * x2(3, 1) * x2(3, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[29] = 2 * d2(0) * d2(3) - d2(0) * d2(0) - d2(3) * d2(3);
    coeffs[30] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(3) * d1(3) * x1(3, 0) * x1(3, 0) + d1(3) * d1(3) * x1(3, 1) * x1(3, 1) -
                 2 * d1(0) * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[31] = d1(0) * d1(0) - 2 * d1(0) * d1(3) + d1(3) * d1(3);
    coeffs[32] = 2 * x2(1, 0) * x2(3, 0) + 2 * x2(1, 1) * x2(3, 1) - x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1) -
                 x2(3, 0) * x2(3, 0) - x2(3, 1) * x2(3, 1);
    coeffs[33] = x1(1, 0) * x1(1, 0) - 2 * x1(1, 1) * x1(3, 1) - 2 * x1(1, 0) * x1(3, 0) + x1(1, 1) * x1(1, 1) +
                 x1(3, 0) * x1(3, 0) + x1(3, 1) * x1(3, 1);
    coeffs[34] = 2 * d2(1) * x2(1, 0) * x2(3, 0) - 2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(3) * x2(3, 0) * x2(3, 0) -
                 2 * d2(3) * x2(3, 1) * x2(3, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * x2(1, 1) * x2(3, 1) +
                 2 * d2(3) * x2(1, 0) * x2(3, 0) + 2 * d2(3) * x2(1, 1) * x2(3, 1);
    coeffs[35] = 2 * d1(1) * x1(1, 0) * x1(1, 0) + 2 * d1(1) * x1(1, 1) * x1(1, 1) + 2 * d1(3) * x1(3, 0) * x1(3, 0) +
                 2 * d1(3) * x1(3, 1) * x1(3, 1) - 2 * d1(1) * x1(1, 0) * x1(3, 0) - 2 * d1(1) * x1(1, 1) * x1(3, 1) -
                 2 * d1(3) * x1(1, 0) * x1(3, 0) - 2 * d1(3) * x1(1, 1) * x1(3, 1);
    coeffs[36] = 2 * d2(1) * d2(3) * x2(1, 0) * x2(3, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                 d2(3) * d2(3) * x2(3, 0) * x2(3, 0) - d2(3) * d2(3) * x2(3, 1) * x2(3, 1) -
                 d2(1) * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * d2(3) * x2(1, 1) * x2(3, 1);
    coeffs[37] = 2 * d2(1) * d2(3) - d2(1) * d2(1) - d2(3) * d2(3);
    coeffs[38] = d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) +
                 d1(3) * d1(3) * x1(3, 0) * x1(3, 0) + d1(3) * d1(3) * x1(3, 1) * x1(3, 1) -
                 2 * d1(1) * d1(3) * x1(1, 0) * x1(3, 0) - 2 * d1(1) * d1(3) * x1(1, 1) * x1(3, 1);
    coeffs[39] = d1(1) * d1(1) - 2 * d1(1) * d1(3) + d1(3) * d1(3);

    const std::vector<int> coeff_ind0 = {
        0,  8,  16, 24, 32, 0,  8,  16, 24, 32, 1,  9,  17, 25, 33, 2,  10, 0,  8,  16, 18, 24, 26, 32, 34, 0,  8,  16,
        24, 32, 1,  9,  17, 25, 33, 2,  10, 8,  0,  16, 18, 24, 26, 32, 34, 8,  0,  16, 24, 32, 1,  9,  17, 25, 33, 3,
        11, 1,  9,  17, 19, 25, 27, 33, 35, 4,  12, 2,  10, 18, 20, 26, 28, 34, 36, 2,  10, 8,  16, 18, 0,  26, 24, 32,
        34, 9,  1,  17, 25, 33, 3,  11, 9,  1,  19, 17, 27, 25, 33, 35, 5,  4,  13, 12, 10, 2,  18, 21, 20, 26, 29, 28,
        34, 36, 37, 10, 2,  8,  0,  18, 24, 26, 32, 34, 16, 3,  11, 19, 9,  17, 1,  27, 25, 33, 35, 6,  14, 3,  11, 19,
        22, 27, 30, 35, 38, 4,  12, 20, 28, 36, 4,  12, 10, 18, 20, 2,  28, 26, 34, 36, 5,  13, 21, 29, 37, 11, 3,  19,
        9,  1,  27, 25, 33, 17, 35, 6,  14, 11, 3,  22, 19, 30, 27, 35, 38, 5,  12, 13, 21, 4,  20, 28, 29, 36, 37, 5,
        12, 13, 4,  10, 21, 2,  20, 26, 29, 28, 34, 36, 37, 18, 7,  15, 23, 31, 39, 6,  14, 22, 11, 19, 3,  30, 27, 35,
        38, 6,  14, 22, 30, 38, 12, 20, 4,  28, 36, 13, 5,  21, 29, 37, 13, 5,  21, 29, 37, 14, 6,  22, 30, 38, 13, 12,
        21, 5,  4,  28, 29, 37, 36, 20, 7,  15, 23, 31, 39, 14, 22, 6,  30, 38, 13, 5,  29, 37, 21, 15, 7,  23, 31, 39,
        7,  15, 23, 31, 39, 14, 6,  22, 11, 3,  30, 27, 35, 19, 38, 7,  15, 23, 31, 39};
    const std::vector<int> coeff_ind1 = {15, 7, 31, 39, 23, 15, 7, 23, 31, 39, 14, 6, 30, 38, 22, 15, 23, 7, 31, 39};
    const std::vector<int> ind0 = {
        0,    2,    18,   28,   39,   41,   45,   60,   69,   78,   80,   82,   98,   108,  119,  120,  122,  123,
        128,  130,  138,  145,  148,  157,  159,  164,  169,  177,  186,  194,  201,  205,  220,  229,  238,  241,
        245,  246,  252,  254,  260,  263,  269,  276,  278,  287,  293,  302,  310,  313,  323,  328,  330,  345,
        357,  360,  362,  364,  369,  377,  378,  386,  388,  394,  399,  400,  402,  403,  408,  410,  418,  425,
        428,  437,  439,  444,  449,  451,  456,  457,  459,  466,  467,  471,  474,  486,  492,  494,  503,  516,
        521,  525,  527,  533,  540,  542,  549,  550,  553,  558,  560,  561,  562,  565,  566,  572,  574,  578,
        580,  583,  588,  589,  596,  598,  599,  607,  613,  615,  621,  622,  624,  630,  632,  633,  635,  643,
        648,  650,  651,  656,  659,  665,  667,  671,  677,  680,  682,  684,  689,  697,  698,  706,  708,  714,
        719,  723,  728,  730,  745,  757,  764,  769,  771,  776,  777,  779,  786,  787,  791,  794,  801,  805,
        820,  829,  838,  846,  852,  854,  855,  861,  863,  864,  872,  875,  876,  881,  885,  887,  893,  900,
        902,  909,  910,  913,  918,  923,  926,  928,  930,  932,  934,  943,  945,  956,  957,  964,  967,  969,
        973,  975,  977,  981,  982,  984,  986,  990,  992,  993,  994,  995,  1000, 1002, 1018, 1028, 1039, 1043,
        1048, 1050, 1051, 1056, 1059, 1065, 1067, 1071, 1077, 1084, 1089, 1097, 1106, 1114, 1131, 1136, 1139, 1147,
        1151, 1166, 1172, 1174, 1183, 1196, 1207, 1213, 1222, 1230, 1233, 1247, 1253, 1262, 1270, 1273, 1291, 1295,
        1296, 1299, 1301, 1304, 1307, 1311, 1312, 1315, 1324, 1329, 1337, 1346, 1354, 1371, 1376, 1379, 1387, 1391,
        1415, 1421, 1424, 1432, 1435, 1446, 1452, 1454, 1463, 1476, 1481, 1485, 1500, 1509, 1518, 1526, 1532, 1534,
        1535, 1541, 1543, 1544, 1552, 1555, 1556, 1563, 1568, 1570, 1585, 1597};
    const std::vector<int> ind1 = {15, 21,  24,  32,  35,  47,  53,  62,  70,  73,
                                   95, 101, 104, 112, 115, 131, 136, 139, 147, 151};
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(40, 40);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(40, 4);

    for (size_t k = 0; k < ind0.size(); k++) {
        int i = ind0[k] % 40;
        int j = ind0[k] / 40;
        C0(i, j) = coeffs[coeff_ind0[k]];
    }

    for (size_t k = 0; k < ind1.size(); k++) {
        int i = ind1[k] % 40;
        int j = ind1[k] / 40;
        C1(i, j) = coeffs[coeff_ind1[k]];
    }

    Eigen::MatrixXd C2 = C0.fullPivHouseholderQr().solve(C1);

    Eigen::Matrix4d AM;
    AM << -C2.row(36), -C2.row(37), -C2.row(38), -C2.row(39);

    Eigen::EigenSolver<Eigen::Matrix4d> es(AM);
    Eigen::Vector4cd D = es.eigenvalues();
    Eigen::Matrix4cd V = es.eigenvectors();

    Eigen::MatrixXcd sols = Eigen::MatrixXcd(4, 5);
    sols.col(0) = (-C2.row(35) * V).array() / V.row(0).array();
    sols.col(1) = D;
    sols.col(2) = V.row(1).array() / V.row(0).array();
    sols.col(3) = V.row(2).array() / V.row(0).array();
    sols.col(4) = V.row(3).array() / V.row(0).array();

    std::vector<Eigen::Vector<double, 6>> solutions;
    for (int i = 0; i < sols.rows(); i++) {
        if (D[i].imag() != 0)
            continue;
        if (sols(i, 3).real() < 0 || sols(i, 4).real() < 0)
            continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector<double, 6> sol;
        sol << 1.0, b1, a2, b2 * a2, 1.0 / std::sqrt(sols(i, 3).real()), 1.0 / std::sqrt(sols(i, 4).real());
        solutions.push_back(sol);
    }


    Eigen::Vector4d depth_x = d1;
    Eigen::Vector4d depth_y = d2;

    for (auto &sol : solutions) {
        Eigen::Vector4d dd1, dd2;

        dd1 = depth_x.array() + sol(1);
        dd2 = depth_y.array() * sol(2) + sol(3);

        if (dd1.minCoeff() <= 0 || dd2.minCoeff() <= 0)
            continue;

        double focal1 = sol(4), focal2 = sol(5);
        Eigen::MatrixXd xu = x1.transpose();
        Eigen::MatrixXd yu = x2.transpose();
        xu.block<2, 4>(0, 0) /= focal1;
        yu.block<2, 4>(0, 0) /= focal2;

        Eigen::MatrixXd X = xu.array().rowwise() * dd1.transpose().array();
        Eigen::MatrixXd Y = yu.array().rowwise() * dd2.transpose().array();

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

        double b2 = sol(1), b1 = sol(3);
        std::swap(b1, b2);

        CameraPose pose = CameraPose(R, t);
        pose.shift_1 = b1;
        pose.shift_2 = b2;
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal1, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
        
    }

}

void varying_focal_monodepth_s00_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                      const std::vector<Eigen::Vector2d> &sigma,
                                      std::vector<ImagePair> *models){
    models->clear();
    models->reserve(1);
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
    double a[18];
    a[0] = x1h[0][0]*depth1[0];
    a[1] = x1h[1][0]*depth1[1];
    a[2] = x1h[2][0]*depth1[2];
    a[3] = x1h[0][1]*depth1[0];
    a[4] = x1h[1][1]*depth1[1];
    a[5] = x1h[2][1]*depth1[2];
    a[6] = depth1[0];
    a[7] = depth1[1];
    a[8] = depth1[2];

    a[9] = x2h[0][0]*depth2[0];
    a[10] = x2h[1][0]*depth2[1];
    a[11] = x2h[2][0]*depth2[2];
    a[12] = x2h[0][1]*depth2[0];
    a[13] = x2h[1][1]*depth2[1];
    a[14] = x2h[2][1]*depth2[2];
    a[15] = depth2[0];
    a[16] = depth2[1];
    a[17] = depth2[2];

    double b[18];
    b[0] = a[0] - a[1];
    b[1] = a[3] - a[4];
    b[2] = a[6] - a[7];
    b[3] = a[0] - a[2];
    b[4] = a[3] - a[5];
    b[5] = a[6] - a[8];
    b[6] = a[1] - a[2];
    b[7] = a[4] - a[5];
    b[8] = a[7] - a[8];
    b[9] = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];
    b[12] = a[9] - a[11];
    b[13] = a[12] - a[14];
    b[14] = a[15] - a[17];
    b[15] = a[10] - a[11];
    b[16] = a[13] - a[14];
    b[17] = a[16] - a[17];

    Eigen::Matrix3d A;
    A << std::pow(b[0], 2) + std::pow(b[1], 2), -std::pow(b[9], 2) - std::pow(b[10], 2), -std::pow(b[11], 2),
        std::pow(b[3], 2) + std::pow(b[4], 2), -std::pow(b[12], 2) - std::pow(b[13], 2), -std::pow(b[14], 2),
        std::pow(b[6], 2) + std::pow(b[7], 2), -std::pow(b[15], 2) - std::pow(b[16], 2), -std::pow(b[17], 2);
    Eigen::Vector3d B;
    B << b[2] * b[2], b[5] * b[5], b[8] * b[8];
    Eigen::Vector3d sol = -A.partialPivLu().solve(B);

    if (sol(0) > 0 && sol(1) > 0 && sol(2) > 0)
    {
        double f = std::sqrt(sol(0));
        double s = std::sqrt(sol(2));
        double w = std::sqrt(sol(1) / sol(2));

        Eigen::Matrix3d K1inv;
        K1inv << f, 0, 0,
            0, f, 0,
            0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << w, 0, 0,
            0,  w, 0,
            0, 0, 1;

        Eigen::Vector3d v1 = s * ((depth2[0]) * K2inv*x2h[0] - (depth2[1]) * K2inv*x2h[1]);
        Eigen::Vector3d v2 = s * ((depth2[0]) * K2inv*x2h[0] - (depth2[2]) * K2inv*x2h[2]);
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * K1inv*x1h[0] - (depth1[1]) * K1inv*x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * K1inv*x1h[0] - (depth1[2]) * K1inv*x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * K1inv*x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * K2inv*x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        double focal1 = 1.0 / f;
        double focal2 = 1.0 / w;

        CameraPose pose = CameraPose(rot, trans);
        pose.scale = s;
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal1, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{focal2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }
}

}
