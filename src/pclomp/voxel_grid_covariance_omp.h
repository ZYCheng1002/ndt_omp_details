/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_VOXEL_GRID_COVARIANCE_OMP_H_
#define PCL_VOXEL_GRID_COVARIANCE_OMP_H_

#include <pcl/filters/boost.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>

#include <map>
#include <unordered_map>

namespace pclomp {
/** \brief A searchable voxel structure containing the mean and covariance of the data.
 * \note For more information please see
 * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
 * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
 * PhD thesis, Orebro University. Orebro Studies in Technology 36</b>
 * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
 */
template <typename PointT>
class VoxelGridCovariance : public pcl::VoxelGrid<PointT> {
 protected:
  using pcl::VoxelGrid<PointT>::filter_name_;
  using pcl::VoxelGrid<PointT>::getClassName;
  using pcl::VoxelGrid<PointT>::input_;  // setInputCloud进行赋值
  using pcl::VoxelGrid<PointT>::indices_;
  using pcl::VoxelGrid<PointT>::filter_limit_negative_;
  using pcl::VoxelGrid<PointT>::filter_limit_min_;
  using pcl::VoxelGrid<PointT>::filter_limit_max_;
  using pcl::VoxelGrid<PointT>::filter_field_name_;

  using pcl::VoxelGrid<PointT>::downsample_all_data_;
  using pcl::VoxelGrid<PointT>::leaf_layout_;
  using pcl::VoxelGrid<PointT>::save_leaf_layout_;
  using pcl::VoxelGrid<PointT>::leaf_size_;
  using pcl::VoxelGrid<PointT>::min_b_;
  using pcl::VoxelGrid<PointT>::max_b_;
  using pcl::VoxelGrid<PointT>::inverse_leaf_size_;  // voxel尺寸的逆
  using pcl::VoxelGrid<PointT>::div_b_;
  using pcl::VoxelGrid<PointT>::divb_mul_;

  typedef typename pcl::traits::fieldList<PointT>::type FieldList;
  typedef typename pcl::Filter<PointT>::PointCloud PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;

 public:
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  typedef pcl::shared_ptr<pcl::VoxelGrid<PointT> > Ptr;
  typedef pcl::shared_ptr<const pcl::VoxelGrid<PointT> > ConstPtr;
#else
  typedef boost::shared_ptr<pcl::VoxelGrid<PointT> > Ptr;
  typedef boost::shared_ptr<const pcl::VoxelGrid<PointT> > ConstPtr;
#endif
  struct Leaf {
    Leaf()
        : nr_points(0),
          mean_(Eigen::Vector3d::Zero()),
          centroid(),
          cov_(Eigen::Matrix3d::Identity()),
          icov_(Eigen::Matrix3d::Zero()),
          evecs_(Eigen::Matrix3d::Identity()),
          evals_(Eigen::Vector3d::Zero()) {}

    /** \brief 返回协方差 */
    Eigen::Matrix3d getCov() const { return (cov_); }

    /** \brief 返回协方差的逆 */
    Eigen::Matrix3d getInverseCov() const { return (icov_); }

    /** \brief 返回体素质心 */
    Eigen::Vector3d getMean() const { return (mean_); }

    /** \brief 返回体素中的特征向量 */
    Eigen::Matrix3d getEvecs() const { return (evecs_); }

    /** \brief 返回体素的特征值 */
    Eigen::Vector3d getEvals() const { return (evals_); }

    /** \brief 返回体素中点数目 */
    int getPointCount() const { return (nr_points); }

    int nr_points;             // 体素中包含的点数
    Eigen::Vector3d mean_;     // 质心点
    Eigen::VectorXf centroid;  // 多维体素质心(当存在color和mean_不同)
    Eigen::Matrix3d cov_;      // 体素中协方差矩阵
    Eigen::Matrix3d icov_;     // 体素中协方差矩阵的逆
    Eigen::Matrix3d evecs_;    // 体素中协方差矩阵的特征向量
    Eigen::Vector3d evals_;    // 体素中协方差矩阵的特征值
  };

  /** \brief Pointer to VoxelGridCovariance leaf structure */
  typedef Leaf* LeafPtr;

  /** \brief Const pointer to VoxelGridCovariance leaf structure */
  typedef const Leaf* LeafConstPtr;

  typedef std::map<size_t, Leaf> Map;

 public:
  /** \brief Constructor.
   * Sets \ref leaf_size_ to 0 and \ref searchable_ to false.
   */
  VoxelGridCovariance()
      : searchable_(true),
        min_points_per_voxel_(6),
        min_covar_eigvalue_mult_(0.01),
        leaves_(),
        voxel_centroids_(),
        voxel_centroids_leaf_indices_(),
        kdtree_() {
    downsample_all_data_ = false;
    save_leaf_layout_ = false;
    leaf_size_.setZero();
    min_b_.setZero();
    max_b_.setZero();
    filter_name_ = "VoxelGridCovariance";
  }

  /** \brief 设置单元格所需的最小点数(为了计算协方差,点数至少为3) */
  inline void setMinPointPerVoxel(int min_points_per_voxel) {
    if (min_points_per_voxel > 2) {
      min_points_per_voxel_ = min_points_per_voxel;
    } else {
      PCL_WARN("%s: Covariance calculation requires at least 3 points, setting Min Point per Voxel to 3 ",
               this->getClassName().c_str());
      min_points_per_voxel_ = 3;
    }
  }

  /** \brief 获取单元格所需的最少点数 */
  inline int getMinPointPerVoxel() { return min_points_per_voxel_; }

  /** \brief Set the minimum allowable ratio between eigenvalues to prevent singular covariance matrices */
  inline void setCovEigValueInflationRatio(double min_covar_eigvalue_mult) {
    min_covar_eigvalue_mult_ = min_covar_eigvalue_mult;
  }

  /** \brief Get the minimum allowable ratio between eigenvalues to prevent singular covariance matrices */
  inline double getCovEigValueInflationRatio() { return min_covar_eigvalue_mult_; }

  /** \brief 点云降采样并初始化体素结构
   * \param[out] output 体素质心点云
   * \param[in] searchable 标记体素是否可被搜索,当时true时建造kdtree
   */
  inline void filter(PointCloud& output, bool searchable = false) {
    searchable_ = searchable;
    applyFilter(output);

    voxel_centroids_ = PointCloudPtr(new PointCloud(output));

    if (searchable_ && voxel_centroids_->size() > 0) {
      /// 使用点云质心构建kdtree
      kdtree_.setInputCloud(voxel_centroids_);
    }
  }

  /** \brief 初始化体素结构
   * \param[in] searchable 标记体素是否可被搜索,当时true时建造kdtree
   */
  inline void filter(bool searchable = false) {
    searchable_ = searchable;
    voxel_centroids_ = PointCloudPtr(new PointCloud);
    applyFilter(*voxel_centroids_);

    if (searchable_ && voxel_centroids_->size() > 0) {
      /// 使用点云质心构建kdtree
      kdtree_.setInputCloud(voxel_centroids_);
    }
  }

  /** \brief Get the voxel containing point p.
   * \param[in] index the index of the leaf structure node
   * \return const pointer to leaf structure
   */
  inline LeafConstPtr getLeaf(int index) {
    auto leaf_iter = leaves_.find(index);
    if (leaf_iter != leaves_.end()) {
      LeafConstPtr ret(&(leaf_iter->second));
      return ret;
    } else
      return NULL;
  }

  /** \brief 获取点p所属的voxel */
  inline LeafConstPtr getLeaf(PointT& p) {
    /// 计算索引
    int ijk0 = static_cast<int>(floor(p.x * inverse_leaf_size_[0]) - min_b_[0]);
    int ijk1 = static_cast<int>(floor(p.y * inverse_leaf_size_[1]) - min_b_[1]);
    int ijk2 = static_cast<int>(floor(p.z * inverse_leaf_size_[2]) - min_b_[2]);

    int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

    /// 查找idx
    auto leaf_iter = leaves_.find(idx);
    if (leaf_iter != leaves_.end()) {
      LeafConstPtr ret(&(leaf_iter->second));
      return ret;
    } else{
      return NULL;
    }
  }

  /** \brief 获取点p所属的voxel */
  inline LeafConstPtr getLeaf(Eigen::Vector3f& p) {
    /// 计算索引
    int ijk0 = static_cast<int>(floor(p[0] * inverse_leaf_size_[0]) - min_b_[0]);
    int ijk1 = static_cast<int>(floor(p[1] * inverse_leaf_size_[1]) - min_b_[1]);
    int ijk2 = static_cast<int>(floor(p[2] * inverse_leaf_size_[2]) - min_b_[2]);

    int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

    auto leaf_iter = leaves_.find(idx);
    if (leaf_iter != leaves_.end()) {
      LeafConstPtr ret(&(leaf_iter->second));
      return ret;
    } else {
      return NULL;
    }
  }

  /** \brief 获取点p周围的体素，不包括包含点p的体素(有效点需要足够)
   * \param[in] reference_point 点
   * \param[out] neighbors
   * \return 数目
   */
  int getNeighborhoodAtPoint(const Eigen::MatrixXi&,
                             const PointT& reference_point,
                             std::vector<LeafConstPtr>& neighbors) const;
  int getNeighborhoodAtPoint(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;
  int getNeighborhoodAtPoint7(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;
  int getNeighborhoodAtPoint1(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;

  /** \brief Get the leaf structure map \return a map containing all leaves */
  inline const Map& getLeaves() { return leaves_; }

  /** \brief 获取质心点
   * \note 数目足够的有效voxel才会返回
   */
  inline PointCloudPtr getCentroids() { return voxel_centroids_; }

  /** \brief 使用云来可视化每个体素的正态分布 */
  void getDisplayCloud(pcl::PointCloud<pcl::PointXYZ>& cell_cloud);

  /** \brief 在给定半径内搜索查询点的所有最近占用体素
   * \note 仅使用包含足够数量的点的体素
   * \param[in] point 查询点
   * \param[in] radius 查询半径
   * \param[out] k_leaves 结果点
   * \param[out] k_sqr_distances 结果索引
   * \param[in] max_nn
   * \return 查寻到的数目
   */
  int nearestKSearch(const PointT& point,
                     int k,
                     std::vector<LeafConstPtr>& k_leaves,
                     std::vector<float>& k_sqr_distances) {
    k_leaves.clear();

    /// Check if kdtree has been built
    if (!searchable_) {
      PCL_WARN("%s: Not Searchable", this->getClassName().c_str());
      return 0;
    }

    /// Find k-nearest neighbors in the occupied voxel centroid cloud
    std::vector<int> k_indices;
    k = kdtree_.nearestKSearch(point, k, k_indices, k_sqr_distances);

    /// 返回voxel
    k_leaves.reserve(k);
    for (std::vector<int>::iterator iter = k_indices.begin(); iter != k_indices.end(); iter++) {
      k_leaves.push_back(&leaves_[voxel_centroids_leaf_indices_[*iter]]);
    }
    return k;
  }

  /** \brief 在给定半径内搜索查询点的所有最近占用体素
   * \note 仅使用包含足够数量的点的体素
   * \param[in] point 查询点
   * \param[in] radius 查询半径
   * \param[out] k_leaves 结果点
   * \param[out] k_sqr_distances 结果索引
   * \param[in] max_nn
   * \return 查寻到的数目
   */
  inline int nearestKSearch(const PointCloud& cloud,
                            int index,
                            int k,
                            std::vector<LeafConstPtr>& k_leaves,
                            std::vector<float>& k_sqr_distances) {
    if (index >= static_cast<int>(cloud.points.size()) || index < 0) return (0);
    return (nearestKSearch(cloud.points[index], k, k_leaves, k_sqr_distances));
  }

  /** \brief 在给定半径内搜索查询点的所有最近占用体素
   * \note 仅使用包含足够数量的点的体素
   * \param[in] point 查询点
   * \param[in] radius 查询半径
   * \param[out] k_leaves 结果点
   * \param[out] k_sqr_distances 结果索引
   * \param[in] max_nn
   * \return 查寻到的数目
   */
  int radiusSearch(const PointT& point,
                   double radius,
                   std::vector<LeafConstPtr>& k_leaves,
                   std::vector<float>& k_sqr_distances,
                   unsigned int max_nn = 0) const {
    k_leaves.clear();

    /// Check if kdtree has been built
    if (!searchable_) {
      PCL_WARN("%s: Not Searchable", this->getClassName().c_str());
      return 0;
    }

    /// Find neighbors within radius in the occupied voxel centroid cloud
    std::vector<int> k_indices;
    int k = kdtree_.radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);

    // Find leaves corresponding to neighbors
    k_leaves.reserve(k);
    for (std::vector<int>::iterator iter = k_indices.begin(); iter != k_indices.end(); iter++) {
      auto leaf = leaves_.find(voxel_centroids_leaf_indices_[*iter]);
      if (leaf == leaves_.end()) {
        std::cerr << "error : could not find the leaf corresponding to the voxel" << std::endl;
        std::cin.ignore(1);
      }
      k_leaves.push_back(&(leaf->second));
    }
    return k;
  }

  /** \brief 在给定半径内搜索查询点的所有最近占用体素
   * \note 仅使用包含足够数量的点的体素
   * \param[in] cloud 查询的点云
   * \param[in] index 需要查询的点云的索引
   * \param[in] radius 查询点半径
   * \param[out] k_leaves 查询点结果
   * \param[out] k_sqr_distances 查询点距离
   * \param[in] max_nn
   * \return 查询到点的数量
   */
  inline int radiusSearch(const PointCloud& cloud,
                          int index,
                          double radius,
                          std::vector<LeafConstPtr>& k_leaves,
                          std::vector<float>& k_sqr_distances,
                          unsigned int max_nn = 0) const {
    if (index >= static_cast<int>(cloud.points.size()) || index < 0) return (0);
    return (radiusSearch(cloud.points[index], radius, k_leaves, k_sqr_distances, max_nn));
  }

 protected:
  /** \brief 过滤并初始化体素结构
   * \param[out] output 包含足够数量点的体素质心的云
   */
  void applyFilter(PointCloud& output);

  bool searchable_;                 // 标记体素是否可搜索
  int min_points_per_voxel_;        // 体素保存的最少点数
  double min_covar_eigvalue_mult_;  // 特征值的比例
  Map leaves_;                      // 包含所有叶节点的体素结构（包括点数不足的体素）
  PointCloudPtr voxel_centroids_;   // 体素网格的质心
  std::vector<int> voxel_centroids_leaf_indices_;  // voxel_centroids的索引
  pcl::KdTreeFLANN<PointT> kdtree_;                // 使用voxel_centroids创建的kdtree
};
}  // namespace pclomp

#endif  // #ifndef PCL_VOXEL_GRID_COVARIANCE_H_
