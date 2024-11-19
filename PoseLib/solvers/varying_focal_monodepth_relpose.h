//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H
#define POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "PoseLib/camera_pose.h"

namespace poselib{
void varying_focal_fundamental_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                       std::vector<ImagePair> *models);

void varying_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                     const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models);
}

#endif // POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H
