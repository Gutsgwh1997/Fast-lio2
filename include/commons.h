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
#include "IKFoM_toolkit/esekfom/esekfom.hpp"

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>

#include "logger.h"

namespace fastlio
{
#define NUM_MATCH_POINTS (5)

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

#define NUM_MAX_POINTS (10000)

    const double G_m_s2 = 9.81;

    extern bool terminate_flag;

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

    typedef MTK::vect<3, double> vect3;
    typedef MTK::SO3<double> SO3;
    typedef MTK::S2<double, 98090, 10000, 1> S2;
    typedef MTK::vect<1, double> vect1;
    typedef MTK::vect<2, double> vect2;

    MTK_BUILD_MANIFOLD(state_ikfom,
                       ((vect3, pos))((SO3, rot))((SO3, offset_R_L_I))((vect3, offset_T_L_I))((vect3, vel))((vect3, bg))((vect3, ba))((S2, grav)));

    MTK_BUILD_MANIFOLD(input_ikfom,
                       ((vect3, acc))((vect3, gyro)));

    MTK_BUILD_MANIFOLD(process_noise_ikfom,
                       ((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));

    MTK::get_cov<process_noise_ikfom>::type process_noise_cov();
    Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in);

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

    struct Pose6D
    {
        Pose6D(int i, double t, Eigen::Matrix3d lr, Eigen::Vector3d lp) : index(i), time(t), local_rot(lr), local_pos(lp) {}
        void setGlobalPose(const Eigen::Matrix3d &gr, const Eigen::Vector3d &gp)
        {
            global_rot = gr;
            global_pos = gp;
        }
        void addOffset(const Eigen::Matrix3d &offset_rot, const Eigen::Vector3d &offset_pos)
        {
            global_rot = offset_rot * local_rot;
            global_pos = offset_rot * local_pos + offset_pos;
        }

        void getOffset(Eigen::Matrix3d &offset_rot, Eigen::Vector3d &offset_pos)
        {
            offset_rot = global_rot * local_rot.transpose();
            offset_pos = -global_rot * local_rot.transpose() * local_pos + global_pos;
        }
        int index;
        double time;
        Eigen::Matrix3d local_rot;
        Eigen::Vector3d local_pos;
        Eigen::Matrix3d global_rot;
        Eigen::Vector3d global_pos;
    };

    struct LoopPair
    {
        LoopPair(int p, int c, float s, Eigen::Matrix3d &dr, Eigen::Vector3d &dp) : pre_idx(p), cur_idx(c), score(s), diff_rot(dr), diff_pos(dp) {}
        int pre_idx;
        int cur_idx;
        Eigen::Matrix3d diff_rot;
        Eigen::Vector3d diff_pos;
        double score;
    };

    struct SharedData
    {
        bool key_pose_added = false;
        std::mutex mutex;
        Eigen::Matrix3d offset_rot = Eigen::Matrix3d::Identity();
        Eigen::Vector3d offset_pos = Eigen::Vector3d::Zero();
        std::vector<Pose6D> key_poses;
        std::vector<LoopPair> loop_pairs;
        std::vector<std::pair<int, int>> loop_history;
        std::vector<fastlio::PointCloudXYZI::Ptr> cloud_history;
    };

}

struct ImuData
{
    std::string topic;
    std::mutex mutex;
    std::deque<fastlio::IMU> buffer;
    double last_timestamp = 0;
    void callback(const sensor_msgs::Imu::ConstPtr &msg);
};

struct LivoxData
{
    std::string topic;
    std::mutex mutex;
    std::deque<fastlio::PointCloudXYZI::Ptr> buffer;
    std::deque<double> time_buffer;
    double blind = 0.5;
    int filter_num = 3;
    double last_timestamp = 0;
    void callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {}
    void livox2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, fastlio::PointCloudXYZI::Ptr &out) {}
};

struct RobosenseM1Data
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        std::uint16_t ring; // 使用 std::uint16_t
        double timestamp;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    std::string topic;
    std::mutex mutex;
    std::deque<fastlio::PointCloudXYZI::Ptr> buffer;
    std::deque<double> time_buffer;
    double blind = 0.6;
    int filter_num = 2;
    double last_timestamp = 0;
    void callback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    bool robosense_m1_2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, fastlio::PointCloudXYZI::Ptr &out, double &s_time);
};
POINT_CLOUD_REGISTER_POINT_STRUCT(RobosenseM1Data::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, timestamp, timestamp))

struct MeasureGroup
{
    double lidar_time_begin = 0.0;
    double lidar_time_end = 0.0;
    bool lidar_pushed = false;
    fastlio::PointCloudXYZI::Ptr lidar;
    std::deque<fastlio::IMU> imus;
    bool syncPackage(ImuData &imu_data, RobosenseM1Data &lidar_data);
};

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

sensor_msgs::PointCloud2 pcl2msg(fastlio::PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp);

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);