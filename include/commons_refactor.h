#pragma once
#include <mutex>
#include <string>
#include <cstdint>
#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>

#include "logger.h"

namespace air_slam
{

#define NUM_MATCH_POINTS (5)

#define MAX_PT_MATCH_DIST (5.0)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

#define NUM_MAX_POINTS (10000)

    const double G_m_s2 = 9.81;

    typedef Eigen::Vector3d V3D;
    typedef Eigen::Matrix3d M3D;
    typedef Eigen::Vector3f V3F;
    typedef Eigen::Matrix3f M3F;

    typedef pcl::PointXYZINormal PointType;
    typedef pcl::PointCloud<PointType> PointCloudXYZI;
    typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

    struct IMU
    {
        IMU() : acc(Eigen::Vector3d::Zero()), gyro(Eigen::Vector3d::Zero()) {}
        IMU(double t, Eigen::Vector3d a, Eigen::Vector3d g)
            : timestamp(t), acc(a), gyro(g) {}
        IMU(double t, double a1, double a2, double a3, double g1, double g2, double g3)
            : timestamp(t), acc(a1, a2, a3), gyro(g1, g2, g3) {}
        double timestamp;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
    };

    bool esti_plane(Eigen::Vector4d &out, const PointVector &points, double thresh);

    float sq_dist(const PointType &p1, const PointType &p2);

    template <typename T, typename Ts>
    Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
    {
        T ang_vel_norm = ang_vel.norm();
        Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

        if (ang_vel_norm > 0.0000001)
        {
            Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
            Eigen::Matrix<T, 3, 3> K;

            K << SKEW_SYM_MATRX(r_axis);

            T r_ang = ang_vel_norm * dt;

            /// Roderigous Tranformation
            return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
        }
        else
        {
            return Eye3;
        }
    }

    struct ImuData
    {
        std::string topic;
        std::mutex mutex;
        std::deque<IMU> buffer;
        double last_timestamp = 0;
        void callback(const sensor_msgs::Imu::ConstPtr &msg);
    };

    struct LivoxData
    {
        std::string topic;
        std::mutex mutex;
        std::deque<PointCloudXYZI::Ptr> buffer;
        std::deque<double> time_buffer;
        double blind = 0.5;
        int filter_num = 3;
        double last_timestamp = 0;
        void callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {}
        void livox2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &out) {}
    };

    struct RobosenseM1Data
    {
        struct EIGEN_ALIGN16 Point
        {
            PCL_ADD_POINT4D;
            float intensity;
            std::uint16_t ring;
            double timestamp;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        std::string topic;
        std::mutex mutex;
        std::deque<PointCloudXYZI::Ptr> buffer;
        std::deque<double> time_buffer; // 激光首点时间戳
        double blind = 0.6;
        int filter_num = 2;
        double last_timestamp = 0;
        void callback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void robosense_m1_2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &out);
    };

    struct MeasureGroup
    {
        double lidar_time_begin = 0.0;
        double lidar_time_end = 0.0;
        bool lidar_pushed = false;
        PointCloudXYZI::Ptr lidar;
        std::deque<IMU> imus;
        bool syncPackage(ImuData &imu_data, RobosenseM1Data &lidar_data);
    };

    nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

    geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

    sensor_msgs::PointCloud2 pcl2msg(PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp);

    Eigen::Vector3d rotate2rpy(const Eigen::Matrix3d &rot);

} // namespace air_slam

POINT_CLOUD_REGISTER_POINT_STRUCT(air_slam::RobosenseM1Data::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, timestamp, timestamp))