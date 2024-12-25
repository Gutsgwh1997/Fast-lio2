#include "commons.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

namespace fastlio {
bool terminate_flag = false;

bool esti_plane(Eigen::Vector4d &out, const PointVector &points, double thresh) {
    Eigen::Matrix<double, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<double, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0;
    for (int i = 0; i < NUM_MATCH_POINTS; i++) {
        A(i, 0) = points[i].x;
        A(i, 1) = points[i].y;
        A(i, 2) = points[i].z;
    }

    Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);

    double norm = normvec.norm();
    out[0] = normvec(0) / norm;
    out[1] = normvec(1) / norm;
    out[2] = normvec(2) / norm;
    out[3] = 1.0 / norm;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (std::fabs(out(0) * points[j].x + out(1) * points[j].y + out(2) * points[j].z + out(3)) > thresh) {
            return false;
        }
    }
    return true;
}

float sq_dist(const PointType &p1, const PointType &p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

MTK::get_cov<process_noise_ikfom>::type process_noise_cov() {
    MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
    MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);  // 0.03
    MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na,
                                                    0.0001);  // *dt 0.01 0.01 * dt * dt 0.05
    MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg,
                                                    0.00001);  // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
    MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba,
                                                    0.00001);  // 0.001 0.05 0.0001/out 0.01
    return cov;
}

Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in) {
    Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    vect3 a_inertial = s.rot * (in.acc - s.ba);
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];
        res(i + 3) = omega[i];
        res(i + 12) = a_inertial[i] + s.grav[i];
    }
    return res;
}

Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in) {
    Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    vect3 acc_;
    in.acc.boxminus(acc_, s.ba);
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
    cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
    Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
    Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
    s.S2_Mx(grav_matrix, vec, 21);
    cov.template block<3, 2>(12, 21) = grav_matrix;
    cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
    return cov;
}

Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in) {
    Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
    cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
    return cov;
}

}  // namespace fastlio

void ImuData::callback(const sensor_msgs::Imu::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mutex);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp) {
        buffer.clear();
        LOG_WARN("imu loop back, clear buffer, last_timestamp: %f  current_timestamp: %f", last_timestamp, timestamp);
    }
    last_timestamp = timestamp;
    buffer.emplace_back(timestamp, msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
                        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
}

void RobosenseM1Data::callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    // rs_m1激光时间戳是尾点时间戳
    std::lock_guard<std::mutex> lock(mutex);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp) {
        buffer.clear();
        time_buffer.clear();
        LOG_WARN("robosense m1 lidat loop back, clear buffer, last_timestamp: %f  current_timestamp: %f",
                 last_timestamp, timestamp);
    }
    last_timestamp = timestamp;
    double s_pt_timestamp = 0.0;
    fastlio::PointCloudXYZI::Ptr ptr(new fastlio::PointCloudXYZI());
    if (robosense_m1_2pcl(msg, ptr, s_pt_timestamp)) {
        buffer.push_back(ptr);
        time_buffer.push_back(s_pt_timestamp);
    }
}

bool RobosenseM1Data::robosense_m1_2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg,
                                        fastlio::PointCloudXYZI::Ptr &out, double &s_time) {
    pcl::PointCloud<RobosenseM1Data::Point> pcl_pts;
    pcl::fromROSMsg(*msg, pcl_pts);
    if (pcl_pts.points.empty()) {
        LOG_WARN("Empty rs m1 lidar points!");
        return false;
    }

    int point_num = pcl_pts.points.size();
    out->clear();
    out->reserve(point_num / filter_num + 1);
    s_time = pcl_pts.points.front().timestamp;

    for (int i = 0; i < point_num; i++) {
        if (i % filter_num != 0) {
            continue;
        }

        const auto &cur_pt = pcl_pts.points[i];
        if (std::isnan(cur_pt.x) || std::isnan(cur_pt.y) || std::isnan(cur_pt.z)) continue;

        double range = cur_pt.x * cur_pt.x + cur_pt.y * cur_pt.y + cur_pt.z * cur_pt.z;
        if (range < (blind * blind)) continue;

        // todo::镜面反射处理
        if (cur_pt.intensity < min_pt_intensity || cur_pt.intensity > max_pt_intensity) continue;

        fastlio::PointType p;
        p.x = cur_pt.x;
        p.y = cur_pt.y;
        p.z = cur_pt.z;
        p.intensity = cur_pt.intensity;
        p.curvature = (cur_pt.timestamp - pcl_pts.points[0].timestamp) * 1000.0;  // s -> ms
        out->push_back(p);
    }
    return true;
}

bool MeasureGroup::syncPackage(ImuData &imu_data, RobosenseM1Data &lidar_data) {
    if (imu_data.buffer.empty() || lidar_data.buffer.empty()) {
        LOG_DEBUG("imu or lidar buffer empty, imu size: %ld, lidar size: %ld", imu_data.buffer.size(),
                  lidar_data.buffer.size());
        return false;
    }
    if (!lidar_pushed) {
        lidar_pushed = true;
        lidar = lidar_data.buffer.front();
        lidar_time_begin = lidar_data.time_buffer.front();
        lidar_time_end = lidar_time_begin + lidar->points.back().curvature / double(1000);
    }
    if (imu_data.last_timestamp < lidar_time_end) {
        LOG_DEBUG("imu blocked, last timestamp: %f, lidar time end: %f", imu_data.last_timestamp, lidar_time_end);
        return false;
    }

    imus.clear();
    double imu_time = imu_data.buffer.front().timestamp;
    while (!imu_data.buffer.empty() && (imu_time < lidar_time_end)) {
        imu_time = imu_data.buffer.front().timestamp;
        if (imu_time > lidar_time_end) break;
        imus.push_back(imu_data.buffer.front());
        imu_data.buffer.pop_front();
    }
    lidar_pushed = false;
    lidar_data.buffer.pop_front();
    lidar_data.time_buffer.pop_front();
    return true;
}

sensor_msgs::PointCloud2 pcl2msg(fastlio::PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*inp, msg);
    if (timestamp < 0)
        msg.header.stamp = ros::Time().now();
    else
        msg.header.stamp = ros::Time().fromSec(timestamp);
    msg.header.frame_id = frame_id;
    return msg;
}

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id,
                                  const std::string &child_frame_id, const double &timestamp) {
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.header.stamp = ros::Time().fromSec(timestamp);
    odom.child_frame_id = child_frame_id;
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    odom.pose.pose.position.x = pos(0);
    odom.pose.pose.position.y = pos(1);
    odom.pose.pose.position.z = pos(2);

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    return odom;
}

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos,
                                                const std::string &frame_id, const std::string &child_frame_id,
                                                const double &timestamp) {
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

Eigen::Vector3d rotate2rpy(const Eigen::Matrix3d &rot) {
    double roll = std::atan2(rot(2, 1), rot(2, 2));
    double pitch = asin(-rot(2, 0));
    double yaw = std::atan2(rot(1, 0), rot(0, 0));

    // 将旋转矩阵转换为欧拉角 (ZYX 顺序)
    // Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);

    return Eigen::Vector3d(roll, pitch, yaw);
}

fastlio::PointCloudXYZI::Ptr addNorm(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZI>);
    searchTree->setInputCloud(cloud);

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(cloud);
    normalEstimator.setSearchMethod(searchTree);
    normalEstimator.setKSearch(15);
    normalEstimator.compute(*normals);
    fastlio::PointCloudXYZI::Ptr out(new fastlio::PointCloudXYZI);
    pcl::concatenateFields(*cloud, *normals, *out);
    return out;
}