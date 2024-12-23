#pragma once

#include <map>
#include <mutex>
#include <vector>
#include <thread>
#include <memory>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "commons.h"
#include "localizer/icp_localizer.h"

namespace fastlio
{
    struct LoopParams
    {
        double rad_thresh = 0.4;
        double dist_thresh = 2.5;
        double time_thresh = 30.0;
        double loop_pose_search_radius = 10.0;
        int loop_pose_index_thresh = 5;
        double submap_resolution = 0.2;
        int submap_search_num = 20;
        double loop_icp_thresh = 0.3;
        bool activate = true;
    };

    class ZaxisPriorFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
    {
    public:
        ZaxisPriorFactor(gtsam::Key key, const gtsam::SharedNoiseModel &noise, double z);
        virtual ~ZaxisPriorFactor();
        virtual gtsam::Vector evaluateError(const gtsam::Pose3 &p, boost::optional<gtsam::Matrix &> H = boost::none) const;

    private:
        double z_;
    };


    class LoopClosureThread
    {
    public:
        void init();

        void operator()();

        void setShared(const std::shared_ptr<SharedData> &share_data);

        void setRate(double rate);

        LoopParams &mutableParams();

        fastlio::PointCloudXYZI::Ptr getSubMaps(std::vector<Pose6D> &pose_list,
                                                std::vector<fastlio::PointCloudXYZI::Ptr> &cloud_list,
                                                int index,
                                                int search_num);
    private:
        void loopCheck();

        void addOdomFactor();

        void addLoopFactor();

        void smoothAndUpdate();

    private:
        double update_rate_ = 1.0;

        std::shared_ptr<SharedData> shared_data_;

        LoopParams loop_params_;

        std::vector<Pose6D> temp_poses_;

        int previous_index_ = 0;

        int lastest_index_;

        bool loop_found_ = false;

        gtsam::Values initialized_estimate_;

        gtsam::Values optimized_estimate_;

        std::shared_ptr<gtsam::ISAM2> isam2_;

        gtsam::NonlinearFactorGraph gtsam_graph_;

        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_history_poses_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_history_poses_;

        pcl::VoxelGrid<fastlio::PointType>::Ptr sub_map_downsize_filter_;

        pcl::IterativeClosestPointWithNormals<fastlio::PointType, fastlio::PointType>::Ptr icp_;
    };

}
