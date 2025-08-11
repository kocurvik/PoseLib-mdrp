# PoseLib implementation for "RePoseD: Efficient Relative Pose Estimation With Known Depth Information"

This is a fork of [PoseLib](https://github.com/PoseLib/PoseLib) with methods evaluated in "RePoseD: Efficient Relative Pose Estimation With Known Depth Information" (ICCV 2025, oral). Pre-print available on Arxiv: [2501.07742](https://arxiv.org/abs/2501.07742)

This repo contains extra code intended for evaluation and ablation experiments. If you are interested only in the novel solvers in clean implementation please refer to the [main project repo](https://github.com/kocurvik/mdrp?tab=readme-ov-file#use-in-your-own-project).

If you want to perform the full evaluation you will need to compile this repository given instructions below and then refer to the [main project repo](https://github.com/kocurvik/mdrp?tab=readme-ov-file#evaluation).

This fork also includes functionality of the original [PoseLib](https://github.com/PoseLib/PoseLib). Refer to its README for more comprehensive documentation.

## How to compile?

Getting the code:

    > git clone --recursive https://github.com/kocurvik/PoseLib-mdrp.git
    > cd PoseLib

Example of a local installation:

    > mkdir _build && cd _build
    > cmake -DCMAKE_INSTALL_PREFIX=../_install ..
    > cmake --build . --target install -j 8
      (equivalent to  'make install -j8' in linux)

Installed files:

    > tree ../_install
      .
      ├── bin
      │   └── benchmark
      ├── include
      │   └── PoseLib
      │       ├── solvers/gp3p.h
      │       ├──  ...
      │       ├── poselib.h          <==  Library header (includes all the rest)
      │       ├──  ...
      │       └── version.h
      └── lib
          ├── cmake
          │   └── PoseLib
          │       ├── PoseLibConfig.cmake
          │       ├── PoseLibConfigVersion.cmake
          │       ├── PoseLibTargets.cmake
          │       └── PoseLibTargets-release.cmake
          └── libPoseLib.a

Uninstall library:

    > make uninstall

## Installation

## Use library (as dependency) in an external project.

    cmake_minimum_required(VERSION 3.13)
    project(Foo)

    find_package(PoseLib REQUIRED)

    add_executable(foo foo.cpp)
    target_link_libraries(foo PRIVATE PoseLib::PoseLib)


## Citing
If you are using methods implemented in this fork please use their respective citations from our paper.

If you use our methods please cite our paper:
```
@inproceedings{ding2025reposed,
  title={RePoseD: Efficient Relative Pose Estimation With Known Depth Information},
  author={Ding, Yaqing and Kocur, Viktor and V{\'a}vra, V{\'a}clav and Haladov{\'a}, Zuzana Berger and Yang, Jian and Sattler, Torsten and Kukelova, Zuzana},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

If you are using PoseLib, please cite the following source:
```
@misc{PoseLib,
  title = {{PoseLib - Minimal Solvers for Camera Pose Estimation}},
  author = {Viktor Larsson and contributors},
  URL = {https://github.com/vlarsson/PoseLib},
  year = {2020}
}
```
Please cite also the original publications of the different methods implemented in original - see table in the original [PoseLib repo](https://github.com/PoseLib/PoseLib).


