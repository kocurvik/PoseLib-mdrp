//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H
#define POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "PoseLib/camera_pose.h"

namespace poselib {
void shared_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                    std::vector<ImagePair> *models);
}

#endif // POSELIB_SHARED_FOCAL_MONODEPTH_RELPOSE_EIGEN_H
