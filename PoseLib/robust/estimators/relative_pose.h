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

#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class SharedFocalRelativePoseEstimator {
  public:
    SharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                     const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;

    const size_t sample_sz = 6;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class SharedFocalMonodepthRelativePoseEstimator {
  public:
    SharedFocalMonodepthRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                              const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &sigma)
        : sample_sz(ransac_opt.use_madpose ? 4 :((ransac_opt.use_ours && ransac_opt.solver_scale && ransac_opt.solver_shift) ? 4 : 3)),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1),
          x2(points2D_2), sigma(sigma),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        monodepth.resize(sample_sz);
        sample.resize(sample_sz);
        x1h.resize(x1.size());
        for (size_t i = 0; i < x1.size(); ++i)
            x1h[i] = x1[i].homogeneous();
        scale_reproj = (opt.max_reproj_error > 0.0) ? (opt.max_epipolar_error * opt.max_epipolar_error) / (opt.max_reproj_error * opt.max_reproj_error) : 0.0;
        weight_sampson = (opt.weight_sampson > 0.0) ? opt.weight_sampson : 0.0;
    }

    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;

    const size_t sample_sz;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &sigma;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector2d> x1s, x2s;
    std::vector<Eigen::Vector3d> x1h;
    std::vector<Eigen::Vector2d> monodepth;
    std::vector<size_t> sample;
    double scale_reproj, weight_sampson;
};

class VaryingFocalMonodepthRelativePoseEstimator {
  public:
    VaryingFocalMonodepthRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                               const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &sigma)
        : sample_sz(ransac_opt.use_fundamental ? 7 : (ransac_opt.use_madpose ? 4 :(ransac_opt.use_ours && (!ransac_opt.solver_scale && !ransac_opt.solver_shift) ? 3 : 4))),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), sigma(sigma),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        monodepth.resize(sample_sz);
        sample.resize(sample_sz);
        x1h.resize(x1.size());
        for (size_t i = 0; i < x1.size(); ++i)
            x1h[i] = x1[i].homogeneous();
    }

    void filter_focals(ImagePairVector *models);
    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;

    const size_t sample_sz = 4;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &sigma;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector2d> x1s, x2s;
    std::vector<Eigen::Vector3d> x1h;
    std::vector<Eigen::Vector2d> monodepth;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class RelativePoseMonoDepthEstimator {
  public:
    RelativePoseMonoDepthEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                         const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &sigma)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), mono_depth(sigma),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x2n.resize(sample_sz);
        X.resize(sample_sz);
        sigmas.resize(sample_sz);
        rel_depth.resize(sample_sz);
        sample.resize(sample_sz);
        if (opt.use_reproj){
            X1.resize(num_data);
            for (size_t i = 0; i < num_data; ++i)
                X1[i] = sigma[i](0) * x1[i].homogeneous();
        }

        scale_reproj = (opt.max_reproj_error > 0.0) ? (opt.max_epipolar_error * opt.max_epipolar_error) / (opt.max_reproj_error * opt.max_reproj_error) : 0.0;
        weight_sampson = (opt.weight_sampson > 0.0) ? opt.weight_sampson : 0.0;
    }
    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;
    const size_t sample_sz = 3;
    const size_t num_data;
  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &mono_depth;
    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector2d> x1s, x2s;
    std::vector<Eigen::Vector2d> sigmas;
    std::vector<Eigen::Vector3d> x2n, X;
    std::vector<double> rel_depth;
    std::vector<size_t> sample;
    std::vector<Eigen::Vector3d> X1;
    double scale_reproj, weight_sampson;
};

} // namespace poselib

#endif