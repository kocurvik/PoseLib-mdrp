//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
#define POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace poselib {

void shared_focal_reldepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                   const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models,
                                   const RansacOptions &opt);

void shared_focal_s00f_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                               const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models);
}

#endif // POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
