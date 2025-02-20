//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H
#define POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace poselib {
void shared_focal_monodepth_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                    std::vector<ImagePair> *models);

void shared_focal_monodepth_3p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &sigma,
                                    std::vector<ImagePair> *models, const RansacOptions &opt);
}

#endif // POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H
