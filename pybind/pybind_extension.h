//
// Created by kocur on 08-Nov-24.
//

#ifndef POSELIB_PYBIND_EXTENSION_H
#define POSELIB_PYBIND_EXTENSION_H

#endif // POSELIB_PYBIND_EXTENSION_H

#include <memory>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// Autocast from numpy.ndarray to std::vector<Eigen::Vector>
namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename Scalar, int Size> struct type_caster<std::vector<Eigen::Matrix<Scalar, Size, 1>>> {
  public:
    using MatrixType = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Size, Eigen::RowMajor>;
    using VectorType = typename Eigen::Matrix<Scalar, Size, 1>;
    using props = EigenProps<MatrixType>;
    PYBIND11_TYPE_CASTER(std::vector<VectorType>, props::descriptor);

    bool load(handle src, bool) {
        const auto buf = array::ensure(src);
        if (!buf) {
            return false;
        }
        const buffer_info info = buf.request();
        if (info.ndim != 2 || info.shape[1] != Size) {
            return false;
        }
        const size_t num_elements = info.shape[0];
        value.resize(num_elements);
        const auto &mat = src.cast<Eigen::Ref<const MatrixType>>();
        Eigen::Map<MatrixType>(reinterpret_cast<Scalar *>(value.data()), num_elements, Size) = mat;
        return true;
    }

    static handle cast(const std::vector<VectorType> &vec, return_value_policy /* policy */, handle h) {
        Eigen::Map<const MatrixType> mat(reinterpret_cast<const Scalar *>(vec.data()), vec.size(), Size);
        return type_caster<Eigen::Map<const MatrixType>>::cast(mat, return_value_policy::copy, h);
    }
};
}}