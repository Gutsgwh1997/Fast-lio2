#pragma once
#include <memory>
#include "commons.h"
#include <Eigen/Geometry>

#include "iekf/esekfom.h"

namespace fastlio
{
    struct Pose
    {
        Pose();
        Pose(double t, Eigen::Vector3d a, Eigen::Vector3d g, Eigen::Vector3d v, Eigen::Vector3d p, Eigen::Matrix3d r)
            : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
        double offset;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        Eigen::Matrix3d rot;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
    };

    class IMUProcessor
    {
    public:
        explicit IMUProcessor(std::shared_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf);

        explicit IMUProcessor(const std::shared_ptr<air_slam::esekfom::esekf>& kf);

        void init(const MeasureGroup &meas);

        void undistortPointcloud(const MeasureGroup &meas, PointCloudXYZI::Ptr &out);

        bool operator()(const MeasureGroup &meas, PointCloudXYZI::Ptr &out);

        bool isInitialized() const { return init_flag_; }

        void setMaxInitCount(int max_init_count) { max_init_count_ = max_init_count; }

        void setExtParams(Eigen::Matrix3d &rot_ext, Eigen::Vector3d &pos_ext);

        void setAccCov(Eigen::Vector3d acc_cov) { acc_cov_ = acc_cov; }

        void setGyroCov(Eigen::Vector3d gyro_cov) { gyro_cov_ = gyro_cov; }

        void setAccBiasCov(Eigen::Vector3d acc_bias_cov) { acc_bias_cov_ = acc_bias_cov; }

        void setGyroBiasCov(Eigen::Vector3d gyro_bias_cov) { gyro_bias_cov_ = gyro_bias_cov; }

        void setCov(Eigen::Vector3d gyro_cov, Eigen::Vector3d acc_cov, Eigen::Vector3d gyro_bias_cov, Eigen::Vector3d acc_bias_cov);

        void setCov(double gyro_cov, double acc_cov, double gyro_bias_cov, double acc_bias_cov);

        void setAlignGravity(bool align_gravity) { align_gravity_ = align_gravity; }

        void reset();

    private:
        int init_count_ = 0;
        int max_init_count_ = 20;
        Eigen::Matrix3d rot_ext_;
        Eigen::Vector3d pos_ext_;
        std::shared_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf_r_;

        std::shared_ptr<air_slam::esekfom::esekf> kf_;

        IMU last_imu_;
        bool init_flag_ = false;
        bool align_gravity_ = true;

        Eigen::Vector3d mean_acc_;
        Eigen::Vector3d mean_gyro_;

        Eigen::Vector3d last_acc_;
        Eigen::Vector3d last_gyro_;

        std::vector<Pose> imu_poses_;

        // 上一帧激光数据的尾点时间
        double last_lidar_time_end_;

        Eigen::Vector3d gyro_cov_;
        Eigen::Vector3d acc_cov_;
        Eigen::Vector3d gyro_bias_cov_;
        Eigen::Vector3d acc_bias_cov_;

        Eigen::Matrix<double, 12, 12> Q_;
    };
} // namespace fastlio
