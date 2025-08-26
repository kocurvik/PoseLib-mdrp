// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "relative_pose.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_6pt_focal.h"
#include "PoseLib/solvers/relpose_7pt.h"
#include "PoseLib/solvers/relpose_mono_3pt.h"
#include "PoseLib/solvers/relpose_reldepth_3pt.h"
#include "PoseLib/solvers/shared_focal_monodepth_relpose_eigen.h"
#include "PoseLib/solvers/shared_focal_reldepth_relpose.h"
#include "PoseLib/solvers/varying_focal_monodepth_relpose.h"

#include <iostream>

namespace poselib {

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_5pt(x1s, x2s, models);
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    if (opt.graduated_steps > 0) {
        bundle_opt.max_iterations = 5;
        for (int k = 0; k < opt.graduated_steps; ++k) {
            double factor = (opt.graduated_steps - k) / static_cast<double>(opt.graduated_steps);
            double tol = opt.max_epipolar_error * 8.0 * factor;
            bundle_opt.loss_scale = tol;
            refine_relpose(x1, x2, pose, bundle_opt);
        }
    }

    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
    refine_relpose(x1, x2, pose, bundle_opt);
}

void SharedFocalRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_6pt_shared_focal(x1s, x2s, models, opt);
}

double SharedFocalRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
    // K_inv << 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    if (opt.graduated_steps > 0) {
        bundle_opt.max_iterations = 5;
        for (int k = 0; k < opt.graduated_steps; ++k) {
            double factor = (opt.graduated_steps - k) / static_cast<double>(opt.graduated_steps);
            double tol = opt.max_epipolar_error * 8.0 * factor;
            bundle_opt.loss_scale = tol;
            refine_shared_focal_relpose(x1, x2, image_pair, bundle_opt);
        }
    }

    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
    refine_shared_focal_relpose(x1, x2, image_pair, bundle_opt);
}

void SharedFocalMonodepthRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]];
        x2s[k] = x2[sample[k]];
        monodepth[k] = sigma[sample[k]];
    }

    if (opt.use_reldepth) {
        shared_focal_reldepth_relpose(x1s, x2s, monodepth, models, opt);
        return;
    }

    if (opt.use_ours) {
        if (opt.solver_scale and opt.solver_shift) {
            shared_focal_monodepth_4p(x1s, x2s, monodepth, opt.use_eigen, models);
            return;
        }

        if (opt.solver_scale and !opt.solver_shift) {
            shared_focal_s00f_relpose(x1s, x2s, monodepth, models);
            return;
        }

        if (!opt.solver_scale and !opt.solver_shift) {
            shared_focal_monodepth_3p(x1s, x2s, monodepth, models, opt);
            return;
        }
    }

    if (opt.use_madpose) {
        shared_focal_monodepth_madpose(x1s, x2s, monodepth, models);
        return;
    }

    throw std::runtime_error("No solver called");
}

double SharedFocalMonodepthRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    if (opt.use_reproj) {
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0 / image_pair.camera1.focal(), 1.0 / image_pair.camera1.focal(), 1.0);
        std::vector<Point3D> X(x1.size());

        if (opt.optimize_shift) {
            double shift = image_pair.pose.shift_1;
            for (size_t i = 0; i < X.size(); ++i) {
                X[i] = (sigma[i](0) + shift) * (K_inv * x1h[i]);
            }
        } else {
            for (size_t i = 0; i < X.size(); ++i) {
                X[i] = sigma[i](0) * (K_inv * x1h[i]);
            }
        }
        return compute_msac_score(image_pair.pose, image_pair.camera1.focal(), x2, X,
                                  opt.max_reproj_error * opt.max_reproj_error, inlier_count);
    }
    Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, image_pair.camera1.focal());
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalMonodepthRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    if (opt.graduated_steps > 0) {
        bundle_opt.max_iterations = 5;
        for (int k = 0; k < opt.graduated_steps; ++k) {
            double factor = (opt.graduated_steps - k) / static_cast<double>(opt.graduated_steps);
            if (opt.use_reproj) {
                double tol = opt.max_reproj_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;
                if (opt.optimize_shift)
                    refine_shared_focal_abspose_shift(x1, x2, sigma, image_pair, bundle_opt);
                else if (opt.optimize_symmetric)
                    refine_shared_focal_symrepro_scale(x1, x2, sigma, image_pair, bundle_opt);
                else
                    refine_shared_focal_abspose(x1, x2, sigma, image_pair, bundle_opt);
            } else {
                double tol = opt.max_epipolar_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;
                refine_shared_focal_relpose(x1, x2, image_pair, bundle_opt);
            }
        }
    }

    bundle_opt.max_iterations = opt.lo_iterations;
    if (opt.use_reproj) {
        bundle_opt.loss_scale = opt.max_reproj_error;
        if (opt.optimize_shift)
            refine_shared_focal_abspose_shift(x1, x2, sigma, image_pair, bundle_opt);
        else if (opt.optimize_symmetric)
            refine_shared_focal_symrepro_scale(x1, x2, sigma, image_pair, bundle_opt);
        else
            refine_shared_focal_abspose(x1, x2, sigma, image_pair, bundle_opt);
    } else {
        bundle_opt.loss_scale = opt.max_epipolar_error;
        refine_shared_focal_relpose(x1, x2, image_pair, bundle_opt);
    }
}

void VaryingFocalMonodepthRelativePoseEstimator::filter_focals(ImagePairVector *models) {
    if (!opt.filter_focals)
        return;

    ImagePairVector new_models;
    new_models.reserve(models->size());

    for (const ImagePair &model : *models) {
        if (model.camera1.focal() >= opt.min_focal_1 and model.camera1.focal() <= opt.max_focal_1 and
            model.camera2.focal() >= opt.min_focal_2 and model.camera2.focal() <= opt.max_focal_2)
            new_models.push_back(model);
    }
    *models = new_models;
}

void VaryingFocalMonodepthRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]];
        x2s[k] = x2[sample[k]];
        monodepth[k] = sigma[sample[k]];
    }

    if (opt.use_fundamental) {
        varying_focal_fundamental_relpose(x1s, x2s, models, opt);
        filter_focals(models);
        return;
    }

    if (opt.use_4p4d) {
        varying_focal_monodepth_relpose(x1s, x2s, monodepth, models, opt);
        filter_focals(models);
        return;
    }

    if (opt.use_ours) {
        if (!opt.solver_scale and !opt.solver_shift) {
            varying_focal_monodepth_abspose_ours(x1s, x2s, monodepth, models, opt);
            filter_focals(models);
            return;
        }

        if (opt.solver_scale and !opt.solver_shift) {
            varying_focal_monodepth_s00_ours(x1s, x2s, monodepth, models);
            filter_focals(models);
            return;
        }

        if (opt.solver_scale and opt.solver_shift) {
            varying_focal_monodepth_relpose_ours(x1s, x2s, monodepth, opt.use_eigen, models);
            filter_focals(models);
            return;
        }
    }

    if (opt.use_madpose) {
        varying_focal_monodepth_relpose_madpose(x1s, x2s, monodepth, models);
        filter_focals(models);
        return;
    }

    throw std::runtime_error("No solver called");
}

double VaryingFocalMonodepthRelativePoseEstimator::score_model(const ImagePair &image_pair,
                                                               size_t *inlier_count) const {
    if (opt.use_reproj) {
        Eigen::DiagonalMatrix<double, 3> K1_inv(1.0 / image_pair.camera1.focal(), 1.0 / image_pair.camera1.focal(),
                                                1.0);
        std::vector<Point3D> X(x1.size());
        if (opt.optimize_shift) {
            double shift = image_pair.pose.shift_1;
            for (size_t i = 0; i < X.size(); ++i) {
                X[i] = (sigma[i](0) + shift) * (K1_inv * x1h[i]);
            }
        } else {
            for (size_t i = 0; i < X.size(); ++i) {
                X[i] = sigma[i](0) * (K1_inv * x1h[i]);
            }
        }

        return compute_msac_score(image_pair.pose, image_pair.camera2.focal(), x2, X,
                                  opt.max_reproj_error * opt.max_reproj_error, inlier_count);

    } else {
        Eigen::DiagonalMatrix<double, 3> K1_inv(1.0, 1.0, image_pair.camera1.focal());
        Eigen::DiagonalMatrix<double, 3> K2_inv(1.0, 1.0, image_pair.camera2.focal());
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d F = K2_inv * (E * K1_inv);

        return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
    }
}

void VaryingFocalMonodepthRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    if (opt.graduated_steps > 0) {
        bundle_opt.max_iterations = 5;
        for (int k = 0; k < opt.graduated_steps; ++k) {
            double factor = (opt.graduated_steps - k) / static_cast<double>(opt.graduated_steps);

            if (opt.use_reproj) {
                double tol = opt.max_reproj_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;

                if (opt.optimize_shift)
                    refine_varying_focal_abspose_shift(x1, x2, sigma, image_pair, bundle_opt);
                else if (opt.optimize_symmetric)
                    refine_varying_focal_symrepro_scale(x1, x2, sigma, image_pair, bundle_opt);
                else
                    refine_varying_focal_abspose(x1, x2, sigma, image_pair, bundle_opt);
            } else {
                double tol = opt.max_epipolar_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;
                refine_varying_focal_relpose(x1, x2, image_pair, bundle_opt);
            }
        }
    }

    bundle_opt.max_iterations = opt.lo_iterations;
    if (opt.use_reproj) {
        bundle_opt.loss_scale = opt.max_reproj_error;
        if (opt.optimize_shift)
            refine_varying_focal_abspose_shift(x1, x2, sigma, image_pair, bundle_opt);
        else if (opt.optimize_symmetric)
            refine_varying_focal_symrepro_scale(x1, x2, sigma, image_pair, bundle_opt);
        else
            refine_varying_focal_abspose(x1, x2, sigma, image_pair, bundle_opt);
    } else {
        bundle_opt.loss_scale = opt.max_epipolar_error;
        refine_varying_focal_relpose(x1, x2, image_pair, bundle_opt);
    }
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // TODO replace by general 6pt solver?

    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_sampson_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error,
                                           &local_inlier_count);
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers with a relaxed threshold
        std::vector<char> inliers;
        int num_inl = get_inliers(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }

    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}

void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);

    if (opt.real_focal_check) {
        for (int i = models->size() - 1; i >= 0; i--) {
            if (!calculate_RFC((*models)[i]))
                models->erase(models->begin() + i);
        }
    }
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

void RelativePoseMonoDepthEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    models->clear();
    if (opt.use_reldepth) {
        for (size_t k = 0; k < sample_sz; ++k) {
            x1s[k] = x1[sample[k]];
            x2s[k] = x2[sample[k]];
            rel_depth[k] = mono_depth[sample[k]](1) / mono_depth[sample[k]](0);
        }
        essential_3pt_relative_depth(x1s, x2s, rel_depth, models, opt.all_permutations);
        return;
    }

    if (opt.use_p3p) {
        for (size_t k = 0; k < sample_sz; ++k) {
            x2n[k] = x2[sample[k]].homogeneous().normalized();
            X[k] = mono_depth[sample[k]](0) * x1[sample[k]].homogeneous();
        }
        p3p(x2n, X, models);
        return;
    }

    if (opt.use_ours and opt.solver_shift and opt.solver_scale) {
        for (size_t k = 0; k < sample_sz; ++k) {
            x1s[k] = x1[sample[k]];
            x2s[k] = x2[sample[k]];
            sigmas[k] = mono_depth[sample[k]];
        }
        essential_3pt_mono_depth(x1s, x2s, sigmas, models, opt.optimize_shift);
        return;
    }

    if (opt.use_madpose) {
        for (size_t k = 0; k < sample_sz; ++k) {
            x1s[k] = x1[sample[k]];
            x2s[k] = x2[sample[k]];
            sigmas[k] = mono_depth[sample[k]];
        }
        essential_3pt_mono_madpose(x1s, x2s, sigmas, models, opt.optimize_shift);
        return;
    }
    throw std::runtime_error("No solver called");
}
double RelativePoseMonoDepthEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
//    if (opt.optimize_hybrid) {
//        return compute_hybrid_msac_score(pose, x1, x2, mono_depth, opt.max_reproj_error * opt.max_reproj_error,
//                                         opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
//    }

    if (opt.use_reproj) {
        if (opt.optimize_shift) {
            std::vector<Point3D> X1s(x1.size());
            double shift = pose.shift_1;
            for (size_t i = 0; i < x1.size(); ++i)
                X1s[i] = (mono_depth[i](0) + shift) * x1[i].homogeneous();
            return compute_msac_score(pose, x2, X1s, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
        }
        return compute_msac_score(pose, x2, X1, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
    }
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}
void RelativePoseMonoDepthEstimator::refine_model(CameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    if (opt.graduated_steps > 0) {
        bundle_opt.max_iterations = 5;
        for (int k = 0; k < opt.graduated_steps; ++k) {
            double factor = (opt.graduated_steps - k) / static_cast<double>(opt.graduated_steps);

            if (opt.optimize_hybrid) {
                // we rescale using scale reproj and scale sampson so set the main loss scale to 1.0
                bundle_opt.loss_scale = 1.0 * 8.0 * factor;
                if (opt.optimize_shift)
                    refine_calib_hybrid_scale_shift(x1, x2, sigmas, pose, scale_reproj, scale_sampson, bundle_opt);
                else
                    refine_calib_hybrid_scale(x1, x2, sigmas, pose, scale_reproj, scale_sampson, bundle_opt);
                continue;
            }

            if (opt.use_reproj) {
                double tol = opt.max_reproj_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;

                if (opt.optimize_symmetric) {
                    refine_calib_symrepro_scale(x1, x2, sigmas, pose, bundle_opt);
                    continue;
                }

                if (opt.optimize_shift) {
                    refine_calib_abspose_shift(x1, x2, sigmas, pose, bundle_opt);
                    continue;
                }

                bundle_adjust(x2, X1, pose, bundle_opt);
                continue;
            } else {
                double tol = opt.max_epipolar_error * 8.0 * factor;
                bundle_opt.loss_scale = tol;
                refine_relpose(x1, x2, pose, bundle_opt);
                continue;
            }
        }
    }

    bundle_opt.max_iterations = opt.lo_iterations;

    if (opt.optimize_hybrid) {
        // we set loss scale to 1.0 since rest is take care or by sacle reproj and scale_sampson
        bundle_opt.loss_scale = 1.0;
        if (opt.optimize_shift)
            refine_calib_hybrid_scale_shift(x1, x2, sigmas, pose, scale_reproj, scale_sampson, bundle_opt);
        else {
            refine_calib_hybrid_scale(x1, x2, sigmas, pose, scale_reproj, scale_sampson, bundle_opt);
        }
        return;
    }

    if (opt.use_reproj) {
        bundle_opt.loss_scale = opt.max_reproj_error;
        if (opt.optimize_symmetric) {
            refine_calib_symrepro_scale(x1, x2, sigmas, pose, bundle_opt);
            return;
        }
        if (opt.optimize_shift) {
            refine_calib_abspose_shift(x1, x2, sigmas, pose, bundle_opt);
            return;
        }
        bundle_adjust(x2, X1, pose, bundle_opt);
        return;
    }
    bundle_opt.loss_scale = opt.max_epipolar_error;
    refine_relpose(x1, x2, pose, bundle_opt);
}
} // namespace poselib
