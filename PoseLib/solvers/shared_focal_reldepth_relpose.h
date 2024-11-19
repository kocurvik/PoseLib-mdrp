//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
#define POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "PoseLib/camera_pose.h"

namespace poselib {

void shared_focal_reldepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                   const std::vector<Eigen::Vector2d> &sigma,
                                   std::vector<ImagePair> *models);
}

#endif // POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
