//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H
#define POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace poselib{
void varying_focal_fundamental_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                       std::vector<ImagePair> *models, const RansacOptions &opt);

void varying_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                     const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models,
                                     const RansacOptions &opt);

void varying_focal_monodepth_relpose_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                          std::vector<ImagePair> *models);

void varying_focal_monodepth_relpose_madpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma, std::vector<ImagePair> *models);

void varying_focal_monodepth_abspose_ours(const std::vector<Eigen::Vector2d> &x1,
                                          const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma,
                                          std::vector<ImagePair> *models,
                                          const RansacOptions &opt);

void varying_focal_monodepth_s00_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                      const std::vector<Eigen::Vector2d> &sigma, bool filter_scale,
                                      std::vector<ImagePair> *models);
}

#endif // POSELIB_VARYING_FOCAL_MONODEPTH_RELPOSE_H
