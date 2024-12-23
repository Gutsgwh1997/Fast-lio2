#include <map>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <ros/ros.h>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "fastlio/SaveMap.h"
#include "lio_builder/lio_builder.h"
#include "lio_builder/loop_closure.h"

#include "commons.h"

namespace fastlio
{
    void signalHandler(int signum)
    {
        terminate_flag = true;
        LOG_WARN("SHUTTING DOWN MAPPING NODE!");
    }

    class MapBuilderROS
    {
    public:
        MapBuilderROS(tf2_ros::TransformBroadcaster &br, const std::shared_ptr<SharedData> &share_data) : br_(br)
        {
            shared_data_ = share_data;
            initPatams();
            initSubscribers();
            initPublishers();
            initServices();

            lio_builder_ = std::make_shared<fastlio::LIOBuilder>(lio_params_);

            loop_closure_.setRate(loop_rate_);
            loop_closure_.setShared(share_data);
            loop_closure_.init();
            loop_thread_ = std::make_shared<std::thread>(std::ref(loop_closure_));
        }

        void initPatams()
        {
            nh_.param<std::string>("map_frame", global_frame_, "map");
            nh_.param<std::string>("local_frame", local_frame_, "local");
            nh_.param<std::string>("body_frame", body_frame_, "body");
            nh_.param<std::string>("imu_topic", imu_data_.topic, "/imu");
            nh_.param<std::string>("rs_m1_topic", rs_m1_data_.topic, "/lidar");

            nh_.param<double>("loop_rate", loop_rate_, 1.0);
            nh_.param<double>("local_rate", local_rate_, 20.0);

            LOG_INFO("Loop Rate: %f.", loop_rate_);
            LOG_INFO("Local Rate: %f.", local_rate_);

            nh_.param<double>("lio_builder/det_range", lio_params_.det_range, 100.0);
            nh_.param<double>("lio_builder/cube_len", lio_params_.cube_len, 500.0);
            nh_.param<double>("lio_builder/resolution", lio_params_.resolution, 0.1);
            nh_.param<double>("lio_builder/move_thresh", lio_params_.move_thresh, 1.5);
            nh_.param<bool>("lio_builder/align_gravity", lio_params_.align_gravity, true);
            nh_.param<std::vector<double>>("lio_builder/imu_ext_rot", lio_params_.imu_ext_rot, std::vector<double>());
            nh_.param<std::vector<double>>("lio_builder/imu_ext_pos", lio_params_.imu_ext_pos, std::vector<double>());

            nh_.param<bool>("loop_closure/activate", loop_closure_.mutableParams().activate, true);
            nh_.param<double>("loop_closure/rad_thresh", loop_closure_.mutableParams().rad_thresh, 0.4);
            nh_.param<double>("loop_closure/dist_thresh", loop_closure_.mutableParams().dist_thresh, 2.5);
            nh_.param<double>("loop_closure/time_thresh", loop_closure_.mutableParams().time_thresh, 30.0);
            nh_.param<double>("loop_closure/loop_pose_search_radius", loop_closure_.mutableParams().loop_pose_search_radius, 10.0);
            nh_.param<int>("loop_closure/loop_pose_index_thresh", loop_closure_.mutableParams().loop_pose_index_thresh, 5);
            nh_.param<double>("loop_closure/submap_resolution", loop_closure_.mutableParams().submap_resolution, 0.2);
            nh_.param<int>("loop_closure/submap_search_num", loop_closure_.mutableParams().submap_search_num, 20);
            nh_.param<double>("loop_closure/loop_icp_thresh", loop_closure_.mutableParams().loop_icp_thresh, 0.3);
        }

        void initSubscribers()
        {
            // todo::buffer缓存应有最大长度限制
            imu_sub_ = nh_.subscribe(imu_data_.topic, 1000, &ImuData::callback, &imu_data_);
            rs_m1_sub_ = nh_.subscribe(rs_m1_data_.topic, 1000, &RobosenseM1Data::callback, &rs_m1_data_);
            LOG_INFO("Subscribe to %s.", imu_data_.topic.c_str());
            LOG_INFO("Subscribe to %s.", rs_m1_data_.topic.c_str());
        }

        void initPublishers()
        {
            local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1000);
            body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
            odom_pub_ = nh_.advertise<nav_msgs::Odometry>("slam_odom", 1000);
            loop_mark_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("loop_mark", 1000);

            local_path_pub_ = nh_.advertise<nav_msgs::Path>("local_path", 1000);
            global_path_pub_ = nh_.advertise<nav_msgs::Path>("global_path", 1000);
        }

        void initServices()
        {
            save_map_server_ = nh_.advertiseService("save_map", &MapBuilderROS::saveMapCallback, this);
        }

        void publishCloud(ros::Publisher &publisher, const sensor_msgs::PointCloud2 &cloud_to_pub)
        {
            if (publisher.getNumSubscribers() == 0)
                return;
            publisher.publish(cloud_to_pub);
        }

        void publishOdom(const nav_msgs::Odometry &odom_to_pub)
        {
            if (odom_pub_.getNumSubscribers() == 0)
                return;
            odom_pub_.publish(odom_to_pub);
        }

        void publishLocalPath()
        {
            if (local_path_pub_.getNumSubscribers() == 0)
                return;

            if (shared_data_->key_poses.empty())
                return;

            nav_msgs::Path path;
            path.header.frame_id = global_frame_;
            path.header.stamp = ros::Time().fromSec(current_time_);
            for (const Pose6D &p : shared_data_->key_poses)
            {
                geometry_msgs::PoseStamped pose;
                pose.header.frame_id = global_frame_;
                pose.header.stamp = ros::Time().fromSec(current_time_);
                pose.pose.position.x = p.local_pos(0);
                pose.pose.position.y = p.local_pos(1);
                pose.pose.position.z = p.local_pos(2);
                Eigen::Quaterniond q(p.local_rot);
                pose.pose.orientation.x = q.x();
                pose.pose.orientation.y = q.y();
                pose.pose.orientation.z = q.z();
                pose.pose.orientation.w = q.w();
                path.poses.push_back(pose);
            }
            local_path_pub_.publish(path);
        }

        void publishGlobalPath()
        {
            if (global_path_pub_.getNumSubscribers() == 0)
                return;

            if (shared_data_->key_poses.empty())
                return;
            nav_msgs::Path path;
            path.header.frame_id = global_frame_;
            path.header.stamp = ros::Time().fromSec(current_time_);
            for (const Pose6D &p : shared_data_->key_poses)
            {
                geometry_msgs::PoseStamped pose;
                pose.header.frame_id = global_frame_;
                pose.header.stamp = ros::Time().fromSec(current_time_);
                pose.pose.position.x = p.global_pos(0);
                pose.pose.position.y = p.global_pos(1);
                pose.pose.position.z = p.global_pos(2);
                Eigen::Quaterniond q(p.global_rot);
                pose.pose.orientation.x = q.x();
                pose.pose.orientation.y = q.y();
                pose.pose.orientation.z = q.z();
                pose.pose.orientation.w = q.w();
                path.poses.push_back(pose);
            }
            global_path_pub_.publish(path);
        }

        void publishLoopMark()
        {
            if (loop_mark_pub_.getNumSubscribers() == 0)
                return;
            if (shared_data_->loop_history.empty())
                return;
            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker nodes_marker;

            nodes_marker.header.frame_id = global_frame_;
            nodes_marker.header.stamp = ros::Time().fromSec(current_time_);
            nodes_marker.ns = "loop_nodes";
            nodes_marker.id = 0;
            nodes_marker.type = visualization_msgs::Marker::SPHERE_LIST;
            nodes_marker.action = visualization_msgs::Marker::ADD;
            nodes_marker.pose.orientation.w = 1.0;
            nodes_marker.scale.x = 0.3;
            nodes_marker.scale.y = 0.3;
            nodes_marker.scale.z = 0.3;
            nodes_marker.color.r = 1.0;
            nodes_marker.color.g = 0.8;
            nodes_marker.color.b = 0.0;
            nodes_marker.color.a = 1.0;

            visualization_msgs::Marker edges_marker;
            edges_marker.header.frame_id = global_frame_;
            edges_marker.header.stamp = ros::Time().fromSec(current_time_);
            edges_marker.ns = "loop_edges";
            edges_marker.id = 1;
            edges_marker.type = visualization_msgs::Marker::LINE_LIST;
            edges_marker.action = visualization_msgs::Marker::ADD;
            edges_marker.pose.orientation.w = 1.0;
            edges_marker.scale.x = 0.1;

            edges_marker.color.r = 0.0;
            edges_marker.color.g = 0.8;
            edges_marker.color.b = 0.0;
            edges_marker.color.a = 1.0;
            for (const auto &p : shared_data_->loop_history)
            {
                const Pose6D &p1 = shared_data_->key_poses[p.first];
                const Pose6D &p2 = shared_data_->key_poses[p.second];
                geometry_msgs::Point point1;
                point1.x = p1.global_pos(0);
                point1.y = p1.global_pos(1);
                point1.z = p1.global_pos(2);
                geometry_msgs::Point point2;
                point2.x = p2.global_pos(0);
                point2.y = p2.global_pos(1);
                point2.z = p2.global_pos(2);
                nodes_marker.points.push_back(point1);
                nodes_marker.points.push_back(point2);
                edges_marker.points.push_back(point1);
                edges_marker.points.push_back(point2);
            }
            marker_array.markers.push_back(nodes_marker);
            marker_array.markers.push_back(edges_marker);
            loop_mark_pub_.publish(marker_array);
        }

        bool saveMapCallback(fastlio::SaveMap::Request &req, fastlio::SaveMap::Response &res)
        {
            float resolution = req.resolution;
            std::string file_path = req.save_path;

            // *保存点云
            bool down_size = fabs(resolution) > 1e-6;
            pcl::VoxelGrid<fastlio::PointType> down_size_filter;
            if (down_size)
            {
                down_size_filter.setLeafSize(resolution, resolution, resolution);
                LOG_INFO("Global map downsize resolution %f.", resolution);
            }
            else
            {
                LOG_WARN("Invalid resolution %f, will not downsize!", resolution);
            }

            fastlio::PointCloudXYZI::Ptr cloud(new fastlio::PointCloudXYZI);
            for (const Pose6D &p : shared_data_->key_poses)
            {
                fastlio::PointCloudXYZI::Ptr temp_cloud(new fastlio::PointCloudXYZI);
                pcl::transformPointCloud(*shared_data_->cloud_history[p.index],
                                         *temp_cloud,
                                         p.global_pos.cast<float>(),
                                         Eigen::Quaternionf(p.global_rot.cast<float>()));
                if (down_size)
                {
                    down_size_filter.setInputCloud(temp_cloud);
                    down_size_filter.filter(*temp_cloud);
                }
                *cloud += *temp_cloud;
            }
            if (cloud->empty())
            {
                res.status = false;
                res.message = "Empty cloud!";
                return false;
            }
            if (down_size)
            {
                down_size_filter.setInputCloud(cloud);
                down_size_filter.filter(*cloud);
            }
            res.status = true;
            res.message = "Save map success!";
            writer_.writeBinaryCompressed(file_path, *cloud);
            LOG_INFO("Save %ld pts to %s success.", cloud->points.size(), file_path.c_str());

            // todo:: 保存轨迹
            return true;
        }

        void run()
        {
            double update_t = 1.0 / (local_rate_ + 1e-6);
            while (ros::ok())
            {
                auto tic = std::chrono::system_clock::now();
                ros::spinOnce();
                if (terminate_flag)
                    break;
                // todo::受限于激光帧于imu帧的时间同步程序设计，后续数据的pub基本为lidar的频率
                if (!measure_group_.syncPackage(imu_data_, rs_m1_data_))
                    continue;
                lio_builder_->mapping(measure_group_);
                if (lio_builder_->currentStatus() == fastlio::Status::INITIALIZE)
                    continue;
                current_time_ = measure_group_.lidar_time_end;
                current_state_ = lio_builder_->currentState();

                br_.sendTransform(eigen2Transform(shared_data_->offset_rot,
                                                  shared_data_->offset_pos,
                                                  global_frame_,
                                                  local_frame_,
                                                  current_time_));
                br_.sendTransform(eigen2Transform(current_state_.rot.toRotationMatrix(),
                                                  current_state_.pos,
                                                  local_frame_,
                                                  body_frame_,
                                                  current_time_));

                publishOdom(eigen2Odometry(current_state_.rot.toRotationMatrix(),
                                           current_state_.pos,
                                           local_frame_,
                                           body_frame_,
                                           current_time_));

                // todo::长距离运行爆内存风险
                addKeyPose();

                publishCloud(body_cloud_pub_,
                             pcl2msg(lio_builder_->cloudUndistortedBody(),
                                     body_frame_,
                                     current_time_));
                publishCloud(local_cloud_pub_,
                             pcl2msg(lio_builder_->cloudWorld(),
                                     local_frame_,
                                     current_time_));
                publishLocalPath();
                publishGlobalPath();
                publishLoopMark();

                auto toc = std::chrono::system_clock::now();
                std::chrono::duration<double> duration = toc - tic;
                if (duration.count() < update_t)
                {
                    std::this_thread::sleep_for(std::chrono::duration<double>(update_t - duration.count()));
                }
            }

            loop_thread_->join();
            LOG_INFO("MAPPING NODE IS DOWN!");
        }

    private:
        void addKeyPose()
        {
            int idx = shared_data_->key_poses.size();
            if (shared_data_->key_poses.empty())
            {
                std::lock_guard<std::mutex> lock(shared_data_->mutex);
                shared_data_->key_poses.emplace_back(idx, current_time_, current_state_.rot.toRotationMatrix(), current_state_.pos);
                shared_data_->key_poses.back().addOffset(shared_data_->offset_rot, shared_data_->offset_pos);
                shared_data_->key_pose_added = true;
                shared_data_->cloud_history.push_back(lio_builder_->cloudUndistortedBody());
                return;
            }

            const Pose6D &last_key_pose = shared_data_->key_poses.back();
            Eigen::Matrix3d diff_rot = last_key_pose.local_rot.transpose() * current_state_.rot.toRotationMatrix();
            Eigen::Vector3d diff_pose = last_key_pose.local_rot.transpose() * (current_state_.pos - last_key_pose.local_pos);
            Eigen::Vector3d rpy = rotate2rpy(diff_rot);
            if (diff_pose.norm() > loop_closure_.mutableParams().dist_thresh ||
                std::abs(rpy(0)) > loop_closure_.mutableParams().rad_thresh ||
                std::abs(rpy(1)) > loop_closure_.mutableParams().rad_thresh ||
                std::abs(rpy(2)) > loop_closure_.mutableParams().rad_thresh)
            {
                std::lock_guard<std::mutex> lock(shared_data_->mutex);
                shared_data_->key_poses.emplace_back(idx, current_time_, current_state_.rot.toRotationMatrix(), current_state_.pos);
                shared_data_->key_poses.back().addOffset(shared_data_->offset_rot, shared_data_->offset_pos);
                shared_data_->key_pose_added = true;
                shared_data_->cloud_history.push_back(lio_builder_->cloudUndistortedBody());
            }
        }

    private:
        ros::NodeHandle nh_;
        std::string global_frame_;
        std::string local_frame_;
        std::string body_frame_;
        double current_time_;
        fastlio::state_ikfom current_state_;
        ImuData imu_data_;
        RobosenseM1Data rs_m1_data_;

        MeasureGroup measure_group_;
        fastlio::LioParams lio_params_;
        std::shared_ptr<fastlio::LIOBuilder> lio_builder_;
        std::shared_ptr<SharedData> shared_data_;
        double local_rate_;
        double loop_rate_;
        LoopClosureThread loop_closure_;
        std::shared_ptr<std::thread> loop_thread_;

        tf2_ros::TransformBroadcaster &br_;

        ros::Subscriber imu_sub_;

        ros::Subscriber rs_m1_sub_;

        ros::Publisher body_cloud_pub_;

        ros::Publisher local_cloud_pub_;

        ros::Publisher odom_pub_;

        ros::Publisher loop_mark_pub_;

        ros::Publisher local_path_pub_;

        ros::Publisher global_path_pub_;

        ros::ServiceServer save_map_server_;

        pcl::PCDWriter writer_;
    };

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_builder_node");
    tf2_ros::TransformBroadcaster br;
    signal(SIGINT, fastlio::signalHandler);

    std::shared_ptr<fastlio::SharedData> share_date = std::make_shared<fastlio::SharedData>();
    fastlio::MapBuilderROS map_builder(br, share_date);
    map_builder.run();
    return 0;
}