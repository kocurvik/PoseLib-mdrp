#ifndef POSELIB_RELPOSE_3PT_MONO_DEPTH_H_
#define POSELIB_RELPOSE_3PT_MONO_DEPTH_H_
#include "PoseLib/camera_pose.h"
#include <Eigen/Dense>
#include <vector>
namespace poselib {
int essential_3pt_mono_depth(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                            const std::vector<Eigen::Vector2d> &sigma, std::vector<CameraPose> *rel_pose);
}; // namespace poselib
#endif