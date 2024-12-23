#include "commons_refactor.h"

namespace air_slam
{
    bool esti_plane(Eigen::Vector4d &out, const PointVector &points, double thresh)
    {
        Eigen::Matrix<double, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<double, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0;
        for (int i = 0; i < NUM_MATCH_POINTS; i++)
        {
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

        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            if (std::fabs(out(0) * points[j].x + out(1) * points[j].y + out(2) * points[j].z + out(3)) > thresh)
            {
                return false;
            }
        }
        return true;
    }

    float sq_dist(const PointType &p1, const PointType &p2)
    {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    void ImuData::callback(const sensor_msgs::Imu::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < last_timestamp)
        {
            buffer.clear();
            LOG_WARN("imu loop back, clear buffer, last_timestamp: %f  current_timestamp: %f", last_timestamp, timestamp);
        }
        last_timestamp = timestamp;
        buffer.emplace_back(timestamp,
                            msg->linear_acceleration.x,
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z,
                            msg->angular_velocity.x,
                            msg->angular_velocity.y,
                            msg->angular_velocity.z);
    }

    void RobosenseM1Data::callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < last_timestamp)
        {
            buffer.clear();
            time_buffer.clear();
            LOG_WARN("robosense m1 lidat loop back, clear buffer, last_timestamp: %f  current_timestamp: %f", last_timestamp, timestamp);
        }
        last_timestamp = timestamp;
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
        robosense_m1_2pcl(msg, ptr);
        buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp);
    }

    void RobosenseM1Data::robosense_m1_2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &out)
    {
        pcl::PointCloud<RobosenseM1Data::Point> pcl_pts;
        pcl::fromROSMsg(*msg, pcl_pts);

        int point_num = pcl_pts.points.size();
        out->clear();
        out->reserve(point_num / filter_num + 1);

        for (int i = 0; i < point_num; i++)
        {
            if (i % filter_num != 0)
                continue;

            const auto &cur_pt = pcl_pts.points[i];
            if (std::isnan(cur_pt.x) || std::isnan(cur_pt.y) || std::isnan(cur_pt.z))
                continue;

            double range = cur_pt.x * cur_pt.x + cur_pt.y * cur_pt.y + cur_pt.z * cur_pt.z;
            if (range < (blind * blind))
                continue;

            PointType p;
            p.x = cur_pt.x;
            p.y = cur_pt.y;
            p.z = cur_pt.z;
            p.intensity = cur_pt.intensity;
            p.curvature = (cur_pt.timestamp - pcl_pts.points[0].timestamp) * 1000.0; // s -> ms
            out->push_back(p);
        }
    }
    bool MeasureGroup::syncPackage(ImuData &imu_data, RobosenseM1Data &lidar_data)
    {
        if (imu_data.buffer.empty() || lidar_data.buffer.empty())
            return false;

        if (!lidar_pushed)
        {
            lidar = lidar_data.buffer.front();
            lidar_time_begin = lidar_data.time_buffer.front();
            lidar_time_end = lidar_time_begin + lidar->points.back().curvature / double(1000);
            lidar_pushed = true;
        }

        if (imu_data.last_timestamp < lidar_time_end)
            return false;
        double imu_time = imu_data.buffer.front().timestamp;
        imus.clear();
        while (!imu_data.buffer.empty() && (imu_time < lidar_time_end))
        {
            imu_time = imu_data.buffer.front().timestamp;
            if (imu_time > lidar_time_end)
                break;
            imus.push_back(imu_data.buffer.front());
            imu_data.buffer.pop_front();
        }
        lidar_data.buffer.pop_front();
        lidar_data.time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    sensor_msgs::PointCloud2 pcl2msg(PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp)
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*inp, msg);
        if (timestamp < 0)
            msg.header.stamp = ros::Time().now();
        else
            msg.header.stamp = ros::Time().fromSec(timestamp);
        msg.header.frame_id = frame_id;
        return msg;
    }

    nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
    {
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

    geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
    {
        geometry_msgs::TransformStamped transform;
        transform.header.frame_id = frame_id;
        transform.header.stamp = ros::Time().fromSec(timestamp);
        transform.child_frame_id = child_frame_id;
        transform.transform.translation.x = pos(0);
        transform.transform.translation.y = pos(1);
        transform.transform.translation.z = pos(2);
        Eigen::Quaterniond q = Eigen::Quaterniond(rot);
        // std::cout << rot << std::endl;
        // std::cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
        transform.transform.rotation.w = q.w();
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        return transform;
    }

    Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot)
    {
        double roll = std::atan2(rot(2, 1), rot(2, 2));
        double pitch = asin(-rot(2, 0));
        double yaw = std::atan2(rot(1, 0), rot(0, 0));
        return Eigen::Vector3d(roll, pitch, yaw);
    }

} // namespace name