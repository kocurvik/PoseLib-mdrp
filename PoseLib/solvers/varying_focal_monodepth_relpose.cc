//
// Created by kocur on 19-Nov-24.
//

#include "varying_focal_monodepth_relpose.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "relpose_7pt.h"

#include <iostream>
namespace poselib {

void varying_focal_fundamental_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                       std::vector<ImagePair> *models) {

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

        if (std::isnan(camera1.focal()))
            return;
        if (std::isnan(camera2.focal()))
            return;

        Eigen::DiagonalMatrix<double, 3> K1(camera1.focal(), camera1.focal(), 1.0);
        Eigen::DiagonalMatrix<double, 3> K2(camera2.focal(), camera2.focal(), 1.0);

        Eigen::Matrix3d E = K2 * F * K1;

        std::vector<CameraPose> poses;
        motion_from_essential(E, x1h, x2h, &poses);

        for (const CameraPose &pose : poses) {
            models->emplace_back(pose, camera1, camera2);
        }
    }
}

void varying_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                     const std::vector<Eigen::Vector2d> &sigma,
                                     std::vector<ImagePair> *models) {
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
        double q = q1 / q2;

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

    if (std::isnan(camera1.focal()))
        return;
    if (std::isnan(camera2.focal()))
        return;

    Eigen::DiagonalMatrix<double, 3> K1(camera1.focal(), camera1.focal(), 1.0);
    Eigen::DiagonalMatrix<double, 3> K2(camera2.focal(), camera2.focal(), 1.0);

    Eigen::Matrix3d E = K2 * F * K1;

    std::vector<CameraPose> poses;
    motion_from_essential(E, x1h, x2h, &poses);

    models->reserve(poses.size());

    for (const CameraPose& pose : poses){
        models->emplace_back(pose, camera1, camera2);
    }
}
}