#include "iekf/esekfom.h"

namespace air_slam
{
    namespace esekfom
    {

        PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));       // 特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
        PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // 有效特征点
        PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 有效特征点对应点法相量
        bool point_selected_surf[100000] = {1};                           // 判断是否是有效特征点
        std::vector<bool> point_selected_surf_buf(100000, true);          // 判断是否是有效特征点

        esekf::esekf() {};
        esekf::~esekf() {};

        const state_ikfom &esekf::get_x() { return x_; }

        esekf::cov esekf::get_P() { return P_; }

        void esekf::change_x(state_ikfom &input_state) { x_ = input_state; }

        void esekf::change_P(cov &input_cov) { P_ = input_cov; }

        // 广义加法  公式(4)
        state_ikfom esekf::boxplus(const state_ikfom &x, const Eigen::Matrix<double, 24, 1> &f)
        {
            state_ikfom x_r;
            x_r.pos = x.pos + f.block<3, 1>(0, 0);

            x_r.rot = x.rot * Sophus::SO3::exp(f.block<3, 1>(3, 0));
            x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3::exp(f.block<3, 1>(6, 0));

            x_r.offset_T_L_I = x.offset_T_L_I + f.block<3, 1>(9, 0);
            x_r.vel = x.vel + f.block<3, 1>(12, 0);
            x_r.bg = x.bg + f.block<3, 1>(15, 0);
            x_r.ba = x.ba + f.block<3, 1>(18, 0);
            x_r.grav = x.grav + f.block<3, 1>(21, 0);

            return x_r;
        }

        // 前向传播  公式(4-8)
        void esekf::predict(double dt, const Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in)
        {
            Eigen::Matrix<double, 24, 1> f_ = get_f(x_, i_in);    // 公式(3)的f
            Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, i_in); // 公式(7)的df/dx
            Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, i_in); // 公式(7)的df/dw

            x_ = boxplus(x_, f_ * dt); // 前向传播 公式(4)

            // 之前Fx矩阵里的项没加单位阵，没乘dt   这里补上
            f_x_ = Matrix<double, 24, 24>::Identity() + f_x_ * dt;

            // 传播协方差矩阵，即公式(8)
            P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose();
        }

        // 计算每个特征点的残差及雅可比矩阵矩阵
        void esekf::h_share_model(dyn_share_datastruct &ekfom_data,
                                  const PointCloudXYZI::Ptr &feats_down_body,
                                  KD_TREE<PointType> &ikdtree,
                                  vector<PointVector> &nearest_points,
                                  bool extrinsic_est)
        {
            laserCloudOri->clear();
            corr_normvect->clear();
            int feats_down_size = feats_down_body->points.size();

#ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
            for (int i = 0; i < feats_down_size; i++) // 遍历所有的特征点
            {
                PointType point_world;
                const PointType &point_body = feats_down_body->points[i];

                // 把Lidar坐标系的点先转到IMU坐标系，再根据前向传播估计的位姿x，转到世界坐标系
                V3D p_body(point_body.x, point_body.y, point_body.z);
                V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
                point_world.x = p_global(0);
                point_world.y = p_global(1);
                point_world.z = p_global(2);
                point_world.intensity = point_body.intensity;

                // nearest_points[i]打印出来发现是按照离point_world距离，从小到大的顺序的vector
                auto &points_near = nearest_points[i];
                vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

                double ta = omp_get_wtime();
                if (ekfom_data.converge)
                {
                    // 寻找point_world的最近邻的平面点
                    ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                    // 判断是否是有效匹配点，与loam系列类似，要求特征点最近邻的地图点数量>阈值，距离<阈值
                    if (points_near.size() >= NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] <= MAX_PT_MATCH_DIST)
                    {
                        point_selected_surf[i] = true;
                    }
                    else
                    {
                        point_selected_surf[i] = false;
                    }
                }
                if (!point_selected_surf[i])
                    continue;

                Eigen::Vector4d pabcd;          // 平面点信息
                point_selected_surf[i] = false; // 将该点设置为无效点，用来判断是否满足条件
                // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
                if (esti_plane(pabcd, points_near, 0.1f))
                {
                    // 当前点到平面的距离
                    float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);

                    // 如果残差大于经验阈值，则认为该点是有效点，距离原点越近的lidar点，要求点到平面的距离越苛刻
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                    if (s > 0.9) // 如果残差大于阈值，则认为该点是有效点
                    {
                        point_selected_surf[i] = true;
                        // 存储平面的单位法向量  以及当前点到平面距离
                        normvec->points[i].x = pabcd(0);
                        normvec->points[i].y = pabcd(1);
                        normvec->points[i].z = pabcd(2);
                        normvec->points[i].intensity = pd2;
                    }
                }
            }

            int effct_feat_num = 0; // 有效特征点的数量
            for (int i = 0; i < feats_down_size; i++)
            {
                if (point_selected_surf[i]) // 对于满足要求的点
                {
                    // 把这些点重新存到laserCloudOri中
                    laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                    // 存储这些点对应的法向量和到平面的距离
                    corr_normvect->points[effct_feat_num] = normvec->points[i];
                    effct_feat_num++;
                }
            }

            if (effct_feat_num < 1)
            {
                ekfom_data.valid = false;
                LOG_WARN("No Effective Points From %d Input Points!", feats_down_size);
                return;
            }

            // 残差向量h和雅可比矩阵h_x的计算
            // H矩阵是稀疏的，只有前12列有非零元素，后12列是零
            ekfom_data.h.resize(effct_feat_num);
            ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
            for (int i = 0; i < effct_feat_num; i++)
            {
                const auto &oriPt = laserCloudOri->points[i];
                const PointType &norm_p = corr_normvect->points[i];
                V3D point(oriPt.x, oriPt.y, oriPt.z);
                V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point);
                V3D point_I = x_.offset_R_L_I * point + x_.offset_T_L_I;
                M3D point_I_crossmat;
                point_I_crossmat << SKEW_SYM_MATRX(point_I);

                // 计算雅可比矩阵H
                V3D C(x_.rot.matrix().transpose() * norm_vec);
                V3D A(point_I_crossmat * C);
                if (extrinsic_est)
                {
                    V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
                        VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                }
                else
                {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
                        VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                }

                // 残差：点面距离
                ekfom_data.h(i) = -norm_p.intensity;
            }
        }

        // 广义减法
        esekf::vectorized_state esekf::boxminus(state_ikfom x1, state_ikfom x2)
        {
            vectorized_state x_r = vectorized_state::Zero();

            x_r.block<3, 1>(0, 0) = x1.pos - x2.pos;

            x_r.block<3, 1>(3, 0) =
                Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
            x_r.block<3, 1>(6, 0) = Sophus::SO3(x2.offset_R_L_I.matrix().transpose() *
                                                x1.offset_R_L_I.matrix())
                                        .log();

            x_r.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
            x_r.block<3, 1>(12, 0) = x1.vel - x2.vel;
            x_r.block<3, 1>(15, 0) = x1.bg - x2.bg;
            x_r.block<3, 1>(18, 0) = x1.ba - x2.ba;
            x_r.block<3, 1>(21, 0) = x1.grav - x2.grav;

            return x_r;
        }

        // ESKF
        void esekf::update_iterated_dyn_share_modified(
            double R,
            const PointCloudXYZI::Ptr &feats_down_body,
            KD_TREE<PointType> &ikdtree,
            vector<PointVector> &nearest_points,
            int maximum_iter,
            bool extrinsic_est)
        {
            nearest_points.resize(feats_down_body->points.size());
            normvec->resize(int(feats_down_body->points.size()));

            dyn_share_datastruct dyn_share;
            dyn_share.valid = true;
            dyn_share.converge = true;
            int t = 0;

            // P_和x_分别是经过正向传播后的协方差矩阵和状态量，会先调用predict再调用这个函数
            cov P_propagated = P_;
            state_ikfom x_propagated = x_;

            vectorized_state dx_new = vectorized_state::Zero(); // 24X1的向量

            // maximum_iter是卡尔曼滤波的最大迭代次数
            for (int i = -1; i < maximum_iter; i++)
            {
                dyn_share.valid = true;
                // 计算点面残差和点面残差的雅克比H(代码里h_x)
                h_share_model(dyn_share, feats_down_body, ikdtree, nearest_points, extrinsic_est);
                if (!dyn_share.valid)
                    continue;

                dx_new = boxminus(x_, x_propagated); // 公式(18)中的 x^k - x^

                // 由于H矩阵是稀疏的，只有前12列有非零元素，后12列是零
                // 因此这里采用分块矩阵的形式计算 减少计算量
                const auto& H = dyn_share.h_x; // m X 12 的矩阵
                Eigen::Matrix<double, 24, 24> HTH = Eigen::Matrix<double, 24, 24>::Zero();
                HTH.block<12, 12>(0, 0) = H.transpose() * H;

                // 公式(14)
                auto K_front = (HTH / R + P_.inverse()).inverse();
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
                K = K_front.block<24, 12>(0, 0) * H.transpose() / R; // 卡尔曼增益这里R视为常数

                Eigen::Matrix<double, 24, 24> KH = Eigen::Matrix<double, 24, 24>::Zero();
                KH.block<24, 12>(0, 0) = K * H;
                // 公式(14)，J_k取单位阵
                vectorized_state dx = K * dyn_share.h + (KH - Eigen::Matrix<double, 24, 24>::Identity()) * dx_new;
                x_ = boxplus(x_, dx); // 公式(18)

                dyn_share.converge = true;
                for (int j = 0; j < 24; j++)
                {
                    if (std::fabs(dx[j]) > epsi) // 如果dx>epsi 认为没有收敛
                    {
                        dyn_share.converge = false;
                        break;
                    }
                }

                if (dyn_share.converge)
                    t++;

                if (!t && i == maximum_iter - 2) // 如果迭代了3次还没收敛,强制令成true，h_share_model函数中会重新寻找近邻点
                {
                    dyn_share.converge = true;
                }

                if (t > 1 || i == maximum_iter - 1)
                {
                    // 公式(15)
                    P_ = (Eigen::Matrix<double, 24, 24>::Identity() - KH) * P_;
                    return;
                }
            }
        }
    } // namespace esekfom
} // namespace air_slam