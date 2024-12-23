#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "commons_refactor.h"
#include "sophus/so3.h"

// 该hpp主要包含：24维状态变量x，输入量u的定义，以及正向传播中相关矩阵的函数

namespace air_slam
{
    namespace esekfom
    {
        // 24维的状态量x
        struct state_ikfom
        {
            Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);
            Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
            Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
            Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d bg = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d grav = Eigen::Vector3d(0, 0, -G_m_s2);
        };

        // 输入u
        struct input_ikfom
        {
            Eigen::Vector3d acc = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d gyro = Eigen::Vector3d(0, 0, 0);
        };

        // 噪声协方差Q的初始化(对应公式(8)的Q, 在IMU_Processing.hpp中使用)
        Eigen::Matrix<double, 12, 12> process_noise_cov();

        // 对应公式(2) 中的f
        Eigen::Matrix<double, 24, 1> get_f(const state_ikfom& s, const input_ikfom& in);

        // 对应公式(7)的Fx  注意该矩阵没乘dt，没加单位阵
        Eigen::Matrix<double, 24, 24> df_dx(const state_ikfom& s, const input_ikfom& in);

        // 对应公式(7)的Fw  注意该矩阵没乘dt
        Eigen::Matrix<double, 24, 12> df_dw(const state_ikfom& s, const input_ikfom& in);
    } // namespace esekfom
} // namespace air_slam

#endif // USE_IKFOM_H1