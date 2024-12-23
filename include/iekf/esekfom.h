#ifndef ESEKFOM_EKF_HPP1
#define ESEKFOM_EKF_HPP1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "iekf/use_ikfom.h"
#include "ikd-Tree/ikd_Tree.h"

// 该hpp主要包含：广义加减法，前向传播主函数，计算特征点残差及其雅可比，ESKF主函数

const double epsi = 0.001; // ESKF迭代时，如果dx<epsi 认为收敛

namespace air_slam
{
    namespace esekfom
    {
        using namespace Eigen;

        extern PointCloudXYZI::Ptr normvec;           // 特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
        extern PointCloudXYZI::Ptr laserCloudOri;     // 有效特征点
        extern PointCloudXYZI::Ptr corr_normvect;     // 有效特征点对应点法相量
        extern bool point_selected_surf[100000];      // 判断是否是有效特征点
        extern std::vector<bool> point_selected_surf_buf; // 判断是否是有效特征点

        struct dyn_share_datastruct
        {
            bool valid;                                                // 有效特征点数量是否满足要求
            bool converge;                                             // 迭代时，是否已经收敛
            Eigen::Matrix<double, Eigen::Dynamic, 1> h;                // 残差	(公式(14)中的z)
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // 雅可比矩阵H (公式(14)中的H)
        };

        class esekf
        {
        public:
            typedef Matrix<double, 24, 24> cov;             // 24X24的协方差矩阵
            typedef Matrix<double, 24, 1> vectorized_state; // 24X1的向量

            esekf();
            ~esekf();

            const state_ikfom &get_x();

            cov get_P();

            void change_x(state_ikfom &input_state);

            void change_P(cov &input_cov);

            // 广义加法  公式(4)
            state_ikfom boxplus(const state_ikfom &x, const Eigen::Matrix<double, 24, 1> &f);

            // 前向传播  公式(4-8)
            void predict(double dt, const Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in);

            // 计算每个特征点的残差及H矩阵
            void h_share_model(dyn_share_datastruct &ekfom_data, const PointCloudXYZI::Ptr &feats_down_body, KD_TREE<PointType> &ikdtree, vector<PointVector> &nearest_points, bool extrinsic_est);

            // 广义减法
            vectorized_state boxminus(state_ikfom x1, state_ikfom x2);

            // ESKF
            void update_iterated_dyn_share_modified(double R, const PointCloudXYZI::Ptr &feats_down_body, KD_TREE<PointType> &ikdtree, vector<PointVector> &nearest_points, int maximum_iter, bool extrinsic_est);

        private:
            state_ikfom x_;
            cov P_ = cov::Identity();
        };
    } // namespace esekfom
} // namespace air_slam

#endif // ESEKFOM_EKF_HPP1