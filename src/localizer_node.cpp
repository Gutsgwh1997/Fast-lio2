#include <nav_msgs/Odometry.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>

#include <chrono>
#include <csignal>
#include <thread>

#include "commons.h"
#include "fastlio/MapConvert.h"
#include "fastlio/SlamHold.h"
#include "fastlio/SlamReLoc.h"
#include "fastlio/SlamRelocCheck.h"
#include "fastlio/SlamStart.h"
#include "lio_builder/lio_builder.h"
#include "localizer/icp_localizer.h"

namespace fastlio {
void signalHandler(int signum) {
    terminate_flag = true;
    LOG_INFO("SHUTTING DOWN LOCALIZER NODE!");
}

struct ReLocSharedData {
    std::mutex main_mutex;
    std::mutex service_mutex;

    bool pose_updated = false;
    bool localizer_activate = false;
    // reloc重定位服务调用标志
    bool service_called = false;
    bool service_success = false;

    std::string map_path;
    Eigen::Matrix3d local_rot;
    Eigen::Vector3d local_pos;
    Eigen::Matrix4d initial_guess;
    fastlio::PointCloudXYZI::Ptr cloud;
    Eigen::Vector3d offset_pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d offset_rot = Eigen::Matrix3d::Identity();

    // 使能系统暂停，lidat与imu数据一直缓存进deque
    bool halt_flag = false;
    bool reset_flag = false;
};

class LocalizerThread {
   public:
    LocalizerThread() {}

    void setSharedDate(const std::shared_ptr<ReLocSharedData> &shared_data) { shared_data_ = shared_data; }

    void setRate(double rate) { rate_ = rate; }

    void setLocalizer(const std::shared_ptr<fastlio::IcpLocalizer> &localizer) { icp_localizer_ = localizer; }

    void operator()() {
        current_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
        std::chrono::duration<double> duration;
        double update_time = 1.0 / (rate_ + 1e-6);
        LOG_INFO("Start relocalizer thread, rate: %f.", rate_);
        while (true) {
            if (duration.count() < update_time) {
                std::this_thread::sleep_for(std::chrono::duration<double>(update_time - duration.count()));
            }
            auto tic = std::chrono::system_clock::now();
            if (terminate_flag) break;
            if (shared_data_->halt_flag) continue;
            if (!shared_data_->localizer_activate) continue;
            if (!shared_data_->pose_updated) continue;

            bool rectify = false;
            gloabl_pose_.setIdentity();
            Eigen::Matrix4d init_guess;
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->pose_updated = false;
                init_guess.setIdentity();
                local_rot_ = shared_data_->local_rot;
                local_pos_ = shared_data_->local_pos;
                init_guess.block<3, 3>(0, 0) = shared_data_->offset_rot * local_rot_;
                init_guess.block<3, 1>(0, 3) = shared_data_->offset_rot * local_pos_ + shared_data_->offset_pos;
                pcl::copyPointCloud(*shared_data_->cloud, *current_cloud_);
            }

            if (shared_data_->service_called) {
                // !调用重定位服务后，首帧重定位成功，localizer_activate，后续单帧ICP
                std::lock_guard<std::mutex> lock(shared_data_->service_mutex);
                shared_data_->service_called = false;
                icp_localizer_->init(shared_data_->map_path, false);
                gloabl_pose_ = icp_localizer_->multi_align_sync(current_cloud_, shared_data_->initial_guess);
                if (icp_localizer_->isSuccess()) {
                    rectify = true;
                    shared_data_->service_success = true;
                    shared_data_->localizer_activate = true;
                    LOG_INFO("Multi_align success, pos_x: %f, pos_y :%f.", gloabl_pose_(0, 3), gloabl_pose_(1, 3));
                } else {
                    rectify = false;
                    shared_data_->service_success = false;
                    shared_data_->localizer_activate = false;
                    LOG_WARN("Multi_align relocalization failed!");
                }
            } else {
                gloabl_pose_ = icp_localizer_->align(current_cloud_, init_guess);
                rectify = icp_localizer_->isSuccess();
            }

            if (rectify) {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                const auto &g_pose = gloabl_pose_;
                shared_data_->offset_rot = g_pose.block<3, 3>(0, 0) * local_rot_.transpose();
                shared_data_->offset_pos = -shared_data_->offset_rot * local_pos_ + g_pose.block<3, 1>(0, 3);
                const auto &o_pos = shared_data_->offset_pos;
                LOG_INFO("Single_align success, pos_x: %f, pos_y: %f.", o_pos[0], o_pos[1]);
            } else {
                LOG_WARN("Single_align relocalization failed!");
            }

            auto toc = std::chrono::system_clock::now();
            duration = toc - tic;
        }
    }

   private:
    double rate_;
    Eigen::Matrix4d gloabl_pose_;
    Eigen::Matrix3d local_rot_;
    Eigen::Vector3d local_pos_;
    std::shared_ptr<ReLocSharedData> shared_data_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_;
};

class LocalizerROS {
   public:
    LocalizerROS(tf2_ros::TransformBroadcaster &br, std::shared_ptr<ReLocSharedData> shared_data)
        : shared_data_(shared_data), br_(br) {
        initParams();
        initSubscribers();
        initPublishers();
        initServices();

        lio_builder_ = std::make_shared<fastlio::LIOBuilder>(lio_params_);
        icp_localizer_ = std::make_shared<fastlio::IcpLocalizer>(
            localizer_params_.refine_resolution, localizer_params_.rough_resolution, localizer_params_.refine_iter,
            localizer_params_.rough_iter, localizer_params_.thresh);
        icp_localizer_->setSearchParams(localizer_params_.xy_offset, localizer_params_.yaw_offset,
                                        localizer_params_.yaw_resolution);

        localizer_loop_.setRate(loop_rate_);
        localizer_loop_.setSharedDate(shared_data_);
        localizer_loop_.setLocalizer(icp_localizer_);
        localizer_thread_ = std::make_shared<std::thread>(std::ref(localizer_loop_));
    }

    void initParams() {
        nh_.param<std::string>("map_frame", global_frame_, "map");
        nh_.param<std::string>("local_frame", local_frame_, "local");
        nh_.param<std::string>("body_frame", body_frame_, "body");
        nh_.param<std::string>("imu_topic", imu_data_.topic, "/imu");
        nh_.param<std::string>("rs_m1_topic", rs_m1_data_.topic, "/lidar");
        nh_.param<bool>("publish_map_cloud", publish_map_cloud_, false);

        nh_.param<double>("loop_rate", loop_rate_, 1.0);
        nh_.param<double>("local_rate", local_rate_, 20.0);

        nh_.param<double>("lio_builder/det_range", lio_params_.det_range, 100.0);
        nh_.param<double>("lio_builder/cube_len", lio_params_.cube_len, 500.0);
        nh_.param<double>("lio_builder/resolution", lio_params_.resolution, 0.1);
        nh_.param<double>("lio_builder/move_thresh", lio_params_.move_thresh, 1.5);
        nh_.param<bool>("lio_builder/align_gravity", lio_params_.align_gravity, true);
        nh_.param<std::vector<double>>("lio_builder/imu_ext_rot", lio_params_.imu_ext_rot, std::vector<double>());
        nh_.param<std::vector<double>>("lio_builder/imu_ext_pos", lio_params_.imu_ext_pos, std::vector<double>());

        nh_.param<double>("localizer/refine_resolution", localizer_params_.refine_resolution, 0.2);
        nh_.param<double>("localizer/rough_resolution", localizer_params_.rough_resolution, 0.5);
        nh_.param<double>("localizer/refine_iter", localizer_params_.refine_iter, 5);
        nh_.param<double>("localizer/rough_iter", localizer_params_.rough_iter, 10);
        nh_.param<double>("localizer/thresh", localizer_params_.thresh, 0.15);

        nh_.param<double>("localizer/xy_offset", localizer_params_.xy_offset, 2.0);
        nh_.param<double>("localizer/yaw_resolution", localizer_params_.yaw_resolution, 0.5);
        nh_.param<int>("localizer/yaw_offset", localizer_params_.yaw_offset, 1);
    }

    void initSubscribers() {
        imu_sub_ = nh_.subscribe(imu_data_.topic, 1000, &ImuData::callback, &imu_data_);
        rs_m1_sub_ = nh_.subscribe(rs_m1_data_.topic, 100, &RobosenseM1Data::callback, &rs_m1_data_);
        LOG_INFO("Subscribe to %s.", imu_data_.topic.c_str());
        LOG_INFO("Subscribe to %s.", rs_m1_data_.topic.c_str());
    }

    void initPublishers() {
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("ego_tf_odom", 1000);
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1000);
        body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        map_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1000);
    }

    bool relocCallback(fastlio::SlamReLoc::Request &req, fastlio::SlamReLoc::Response &res) {
        std::string map_path = req.pcd_path;
        float x = req.x;
        float y = req.y;
        float z = req.z;
        float roll = req.roll;
        float pitch = req.pitch;
        float yaw = req.yaw;
        Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = rollAngle * pitchAngle * yawAngle;
        {
            std::lock_guard<std::mutex> lock(shared_data_->service_mutex);
            shared_data_->halt_flag = false;
            shared_data_->service_called = true;
            shared_data_->localizer_activate = true;
            shared_data_->map_path = map_path;
            shared_data_->initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix().cast<double>();
            shared_data_->initial_guess.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
        }
        res.status = 1;
        res.message = "RELOCALIZE CALLED!";

        return true;
    }

    bool mapConvertCallback(fastlio::MapConvert::Request &req, fastlio::MapConvert::Response &res) {
        pcl::PCDReader reader;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        reader.read(req.map_path, *cloud);
        pcl::VoxelGrid<pcl::PointXYZI> down_sample_filter;
        down_sample_filter.setLeafSize(req.resolution, req.resolution, req.resolution);
        down_sample_filter.setInputCloud(cloud);
        down_sample_filter.filter(*cloud);

        fastlio::PointCloudXYZI::Ptr cloud_with_norm = addNorm(cloud);
        pcl::PCDWriter writer;
        writer.writeBinaryCompressed(req.save_path, *cloud_with_norm);
        res.message = "CONVERT SUCCESS!";
        res.status = 1;

        return true;
    }

    bool slamHoldCallback(fastlio::SlamHold::Request &req, fastlio::SlamHold::Response &res) {
        shared_data_->service_mutex.lock();
        shared_data_->halt_flag = true;
        shared_data_->reset_flag = true;
        shared_data_->service_mutex.unlock();
        res.message = "SLAM HALT!";
        res.status = 1;
        return true;
    }

    bool slamStartCallback(fastlio::SlamStart::Request &req, fastlio::SlamStart::Response &res) {
        shared_data_->service_mutex.lock();
        shared_data_->halt_flag = false;
        shared_data_->service_mutex.unlock();
        res.message = "SLAM START!";
        res.status = 1;
        return true;
    }

    bool slamRelocCheckCallback(fastlio::SlamRelocCheck::Request &req, fastlio::SlamRelocCheck::Response &res) {
        res.status = shared_data_->service_success;
        return true;
    }

    void initServices() {
        reloc_server_ = nh_.advertiseService("slam_reloc", &LocalizerROS::relocCallback, this);
        map_convert_server_ = nh_.advertiseService("map_convert", &LocalizerROS::mapConvertCallback, this);
        hold_server_ = nh_.advertiseService("slam_hold", &LocalizerROS::slamHoldCallback, this);
        start_server_ = nh_.advertiseService("slam_start", &LocalizerROS::slamStartCallback, this);
        reloc_check_server_ = nh_.advertiseService("slam_reloc_check", &LocalizerROS::slamRelocCheckCallback, this);
    }

    void publishCloud(ros::Publisher &publisher, const sensor_msgs::PointCloud2 &cloud_to_pub) {
        if (publisher.getNumSubscribers() == 0) return;
        publisher.publish(cloud_to_pub);
    }

    void publishOdom(const nav_msgs::Odometry &odom_to_pub) {
        if (odom_pub_.getNumSubscribers() == 0) return;
        odom_pub_.publish(odom_to_pub);
    }

    void systemReset() {
        offset_pos_ = Eigen::Vector3d::Zero();
        offset_rot_ = Eigen::Matrix3d::Identity();
        {
            std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
            shared_data_->offset_rot = Eigen::Matrix3d::Identity();
            shared_data_->offset_pos = Eigen::Vector3d::Zero();
            shared_data_->service_success = false;
        }
        lio_builder_->reset();
    }

    void run() {
        std::chrono::duration<double> duration;
        double update_time = 1.0 / (local_rate_ + 1e-6);
        LOG_INFO("Start main thread, rate: %f.", local_rate_);
        while (ros::ok()) {
            if (duration.count() < update_time) {
                std::this_thread::sleep_for(std::chrono::duration<double>(update_time - duration.count()));
            }
            auto tic = std::chrono::system_clock::now();

            ros::spinOnce();
            if (terminate_flag) {
                break;
            }
            if (!measure_group_.syncPackage(imu_data_, rs_m1_data_)) {
                continue;
            }

            if (shared_data_->halt_flag) {
                continue;
            }
            if (shared_data_->reset_flag) {
                LOG_INFO("SLAM RESET!");
                systemReset();
                shared_data_->service_mutex.lock();
                shared_data_->reset_flag = false;
                shared_data_->service_mutex.unlock();
            }

            lio_builder_->mapping(measure_group_);
            if (lio_builder_->currentStatus() == fastlio::Status::INITIALIZE) {
                continue;
            }
            current_time_ = measure_group_.lidar_time_end;
            current_state_ = lio_builder_->currentState();
            current_cloud_body_ = lio_builder_->cloudUndistortedBody();
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->local_rot = current_state_.rot.toRotationMatrix();
                shared_data_->local_pos = current_state_.pos;
                shared_data_->cloud = current_cloud_body_;
                offset_rot_ = shared_data_->offset_rot;
                offset_pos_ = shared_data_->offset_pos;
                shared_data_->pose_updated = true;
            }

            auto rot_mat = current_state_.rot.toRotationMatrix();
            br_.sendTransform(eigen2Transform(offset_rot_, offset_pos_, global_frame_, local_frame_, current_time_));
            br_.sendTransform(eigen2Transform(rot_mat, current_state_.pos, local_frame_, body_frame_, current_time_));
            publishOdom(eigen2Odometry(rot_mat, current_state_.pos, local_frame_, body_frame_, current_time_));
            publishCloud(body_cloud_pub_, pcl2msg(current_cloud_body_, body_frame_, current_time_));
            publishCloud(local_cloud_pub_, pcl2msg(lio_builder_->cloudWorld(), local_frame_, current_time_));

            if (publish_map_cloud_) {
                if (icp_localizer_->isInitialized()) {
                    publishCloud(map_cloud_pub_, pcl2msg(icp_localizer_->getRoughMap(), global_frame_, current_time_));
                }
            }

            auto toc = std::chrono::system_clock::now();
            duration = toc - tic;
        }

        localizer_thread_->join();
        LOG_INFO("LOCALIZER NODE IS DOWN!");
    }

   private:
    double loop_rate_;
    double local_rate_;
    std::string body_frame_;
    std::string local_frame_;
    std::string global_frame_;

    double current_time_;
    bool publish_map_cloud_;

    ImuData imu_data_;
    RobosenseM1Data rs_m1_data_;
    MeasureGroup measure_group_;
    fastlio::state_ikfom current_state_;
    fastlio::PointCloudXYZI::Ptr current_cloud_body_;
    std::shared_ptr<fastlio::LIOBuilder> lio_builder_;

    fastlio::LioParams lio_params_;
    LocalizerThread localizer_loop_;
    std::shared_ptr<ReLocSharedData> shared_data_;
    fastlio::RelocalizerParams localizer_params_;
    std::shared_ptr<std::thread> localizer_thread_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;

    Eigen::Vector3d offset_pos_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d offset_rot_ = Eigen::Matrix3d::Identity();

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
    ros::Subscriber rs_m1_sub_;
    ros::Publisher odom_pub_;
    ros::Publisher body_cloud_pub_;
    ros::Publisher local_cloud_pub_;
    ros::Publisher map_cloud_pub_;
    ros::ServiceServer reloc_server_;
    ros::ServiceServer map_convert_server_;
    ros::ServiceServer reloc_check_server_;
    ros::ServiceServer hold_server_;
    ros::ServiceServer start_server_;
    tf2_ros::TransformBroadcaster &br_;
};

}  // namespace fastlio

int main(int argc, char **argv) {
    ros::init(argc, argv, "localizer_node");
    tf2_ros::TransformBroadcaster br;
    signal(SIGINT, fastlio::signalHandler);
    std::shared_ptr<fastlio::ReLocSharedData> shared_date = std::make_shared<fastlio::ReLocSharedData>();
    fastlio::LocalizerROS localizer_ros(br, shared_date);
    localizer_ros.run();
    return 0;
}