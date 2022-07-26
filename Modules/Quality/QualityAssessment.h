#pragma once
#include <math.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PCLPointCloud2.h>
//#include <pcl/filters/voxel_grid.h>
//#include <iostream>
using namespace std;



struct Point_3d_lokal
{
	float std_lokal;
	PCL_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_3d_lokal,          
	(float, std_lokal, std_lokal)
)
// Berechnung der Standardabweichung der Distanzmessung -> Intensitätswerte dürfen nicht normiert sein
// Calculation of the precission of the distance measurement -> Intensity values needs to be raw, not scaled
void sigma_distance(float& sigma_dist, const float& a, const float& b, const float& c,const float& intens ) {
	sigma_dist = a * pow(intens, b) + c;
}


//// Variance covariance propagation 
// sigma_3D_Pos = 3D precision 
// Sigma_XX2 = Variance Covariance Matrix of parameters (X,Y,Z)
// Sigma_LL = Variance Covariance Matrix of observations (H,V,S)
//
template <typename PointT>
void sigma_3D_Position(float& sigma_3D_pos, Eigen::Matrix3f& Sigma_XX2 ,Eigen::Matrix3f& Sigma_LL, PointT& Point) {

	float dist = sqrt(pow(Point.x, 2) + pow(Point.y, 2) + pow(Point.z, 2));
	float az = atan2(Point.y, Point.x);
	float elev = atan2(sqrt(pow(Point.x, 2) + pow(Point.y, 2)), Point.z); // Umgestellt

	Eigen::Matrix3f A; // Designmatrix
	A(0, 0) = -dist * cos(elev) * sin(az);
	A(0, 1) = -dist * cos(az) * sin(elev);
	A(0, 2) = cos(elev) * cos(az);
	A(1, 0) = dist * cos(elev) * cos(az);
	A(1, 1) = -dist * sin(elev) * sin(az);
	A(1, 2) = cos(elev) * sin(az);
	A(2, 0) = 0;
	A(2, 1) = -dist * sin(elev);
	A(2, 2) = cos(elev);

	Sigma_XX2 = A * Sigma_LL * A.transpose();
	//float sx = sqrt(Sigma_XX2(0, 0));
	//float sy = sqrt(Sigma_XX2(1, 1));
	//float sz = sqrt(Sigma_XX2(2, 2));
	sigma_3D_pos = sqrt(Sigma_XX2(0, 0) + Sigma_XX2(1, 1) + Sigma_XX2(2, 2));
}



// Calculation of the local 3D precision for the whole point cloud -> parallel computing using omp
// cloud_result = "vector" containing all the 3D precisions
// cloud = input cloud with intensities
// a,b,c = parameters of intensity based model
// sigma_h, sigma_v = standard deviation of angle measurement
//linear_error = linearity error given by the manufacturer (if not known, it should be set to 0)
template <typename PointT>
void calculate_3D_std_lokal(pcl::PointCloud<Point_3d_lokal>&cloud_result,const pcl::PointCloud<PointT>& cloud,const float&a,const float&b,const float& c, const float&sigma_h, const float&sigma_v, const float& linear_error){
	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	cloud_result.sensor_origin_ = cloud.sensor_origin_;
	cloud_result.sensor_orientation_ = cloud.sensor_orientation_;
int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
for (int i = 0; i < cloud.size() - 1; i++) {
	float std_dist;
	float std_lokal;
	sigma_distance(std_dist, a, b, c, cloud.points[i].intensity);
	Eigen::Matrix3f Sigma_XX = Eigen::Matrix3f::Zero(); // VKV der Parameter
	Eigen::Matrix3f Sigma_LL = Eigen::Matrix3f::Zero(); // VKV der Beobachtungen
	Sigma_LL(0, 0) = sigma_h * sigma_h;
	Sigma_LL(1, 1) = sigma_v * sigma_v;
	Sigma_LL(2, 2) = std_dist * std_dist + linear_error*linear_error;
	//Sigma_LL(2, 2) = Sigma_LL(2, 2) + pow(0.00043, 2);
	sigma_3D_Position(std_lokal, Sigma_XX, Sigma_LL, cloud.points[i]);
	//if (isinf(std_dist)) {
	//
	//	std_dist = NAN;
	//}
	cloud_result.points[i].std_lokal = std_lokal;
	}


}


/// Global 3D precision
// sigma_3D_transf = 3D global precision
// searchPoint = Point for calculation
// Quat = Quaternion of the transformation (Rotation)
// Sigma_LL_Transf = VCV (7x7) of observation (4 quaternions, 3 translations). Order is: q0,q1,q2,q3,x,y,z
//Sigma_LL_Pos = VCV of parameters (X,Y,Z)_global

template <typename PointT>
void sigma_3D_Transformation(float& sigma_3D_transf, const PointT& searchPoint, const Eigen::Quaternionf& Quat, const Eigen::MatrixXf& Sigma_LL_Transf, const Eigen::Matrix3f Sigma_LL_pos) {

	Eigen::MatrixXf Sigma_LL_ALL = Eigen::MatrixXf::Zero(10, 10);

	Sigma_LL_ALL.topLeftCorner(7, 7) = Sigma_LL_Transf;

	Sigma_LL_ALL.bottomRightCorner(3, 3) = Sigma_LL_pos;

	float q0 = Quat.w();
	float q1 = Quat.x();
	float q2 = Quat.y();
	float q3 = Quat.z();
	Eigen::MatrixXf F(3, 10);

	F(0, 0) = 4 * q0 * searchPoint.x + 2 * q3 * searchPoint.y - 2 * q2 * searchPoint.z;
	F(0, 1) = 4 * q1 * searchPoint.x + 2 * q2 * searchPoint.y + 2 * q3 * searchPoint.z;
	F(0, 2) = 2 * q1 * searchPoint.y - 2 * q0 * searchPoint.z;
	F(0, 3) = 2 * q0 * searchPoint.y + 2 * q1 * searchPoint.z;
	F(0, 4) = 1;
	F(0, 5) = 0;
	F(0, 6) = 0;
	F(0, 7) = 2 * pow(q0, 2) + 2 * pow(q1, 2) - 1;
	F(0, 8) = 2 * q0 * q3 + 2 * q1 * q2;
	F(0, 9) = 2 * q1 * q3 - 2 * q0 * q2;


	F(1, 0) = 4 * q0 * searchPoint.y - 2 * q3 * searchPoint.x + 2 * q1 * searchPoint.z;
	F(1, 1) = 2 * q2 * searchPoint.x + 2 * q0 * searchPoint.z;
	F(1, 2) = 2 * q1 * searchPoint.x + 4 * q2 * searchPoint.y + 2 * q3 * searchPoint.z;
	F(1, 3) = 2 * q2 * searchPoint.z - 2 * q0 * searchPoint.x;
	F(1, 4) = 0;
	F(1, 5) = 1;
	F(1, 6) = 0;
	F(1, 7) = 2 * q1 * q2 - 2 * q0 * q3;
	F(1, 8) = 2 * pow(q0, 2) + 2 * pow(q2, 2) - 1;
	F(1, 9) = 2 * q0 * q1 + 2 * q2 * q3;


	F(2, 0) = 2 * q2 * searchPoint.x - 2 * q1 * searchPoint.y + 4 * q0 * searchPoint.z;
	F(2, 1) = 2 * q3 * searchPoint.x - 2 * q0 * searchPoint.y;
	F(2, 2) = 2 * q0 * searchPoint.x + 2 * q3 * searchPoint.y;
	F(2, 3) = 2 * q1 * searchPoint.x + 2 * q2 * searchPoint.y + 4 * q3 * searchPoint.z;
	F(2, 4) = 0;
	F(2, 5) = 0;
	F(2, 6) = 1;
	F(2, 7) = 2 * q0 * q2 + 2 * q1 * q3;
	F(2, 8) = 2 * q2 * q3 - 2 * q0 * q1;
	F(2, 9) = 2 * pow(q0, 2) + 2 * pow(q3, 2) - 1;

	Eigen::Matrix3f Sigma_XX;
	Sigma_XX = F * Sigma_LL_ALL * F.transpose();


	float sx = sqrt(Sigma_XX(0, 0));
	float sy = sqrt(Sigma_XX(1, 1));
	float sz = sqrt(Sigma_XX(2, 2));
	sigma_3D_transf = sqrt(pow(sx, 2) + pow(sy, 2) + pow(sz, 2));

}





struct Point_3d_global
{
	float std_3D_global;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_3d_global,           // here we assume a XYZ + "test" (as fields)
	(float, std_3D_global, std_3D_global)
)
///
//// Calculation of the global 3D precision for the whole point cloud -> parallel computing using omp
// cloud_result = "vector" containing all the 3D global precisions
// glob = true -> use VCV from local precision || glob = false -> only propagate uncertainties coming from the Transformation
// Quat = Quaternion of the Transformation (Rotation)
// Sigma_LL_Transf = VCV (7x7) of observation (4 quaternions, 3 translations). Order is: q0,q1,q2,q3,x,y,z
// a,b,c = parameters of intensity based model
// sigma_h, sigma_v = standard deviation of angle measurement
//linear_error = linearity error given by the manufacturer (if not known, it should be set to 0)
template <typename PointT>  void
Pcl_std_3D_global(pcl::PointCloud<Point_3d_global>& cloud_result,const pcl::PointCloud<PointT>& cloud, bool& glob, const Eigen::Quaternionf& Quat, const Eigen::MatrixXf& Sigma_LL_Transf, const float& sigma_h, const float& sigma_v, const float& a, const float& b, float& c, float linear_error) {
	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());

	int cnt = omp_get_num_procs();
	#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i < cloud.size() - 1; i++) {
		float std_dist = 0;
		float std_lokal = 0;
		float std_global;
		Eigen::Matrix3f Sigma_XX = Eigen::Matrix3f::Zero(); // VKV der Parameter
		if (glob == true) {
			sigma_distance(std_dist, a, b, c, cloud.points[i].intensity);
			
			Eigen::Matrix3f Sigma_LL = Eigen::Matrix3f::Zero(); // VKV der Beobachtungen
			Sigma_LL(0, 0) = sigma_h * sigma_h;
			Sigma_LL(1, 1) = sigma_v * sigma_v;
			Sigma_LL(2, 2) = std_dist * std_dist + linear_error * linear_error;
			//Sigma_LL(2, 2) = Sigma_LL(2, 2) + pow(0.00043, 2);
			sigma_3D_Position(std_lokal, Sigma_XX, Sigma_LL, cloud.points[i]);
		}

		
		Eigen::Matrix3f Sigma_LL_pos;
		sigma_3D_Transformation(std_global, cloud.points[i],Quat, Sigma_LL_Transf, Sigma_XX);
		cloud_result.points[i].std_3D_global = std_global;
	}
}


// Incidence angle calculation
// inc = incidence angle
// n = normalvector
// search Point = points for which the incidence angle is calculated
template <typename PointT>void
Incidence_angle(float& inc, const Eigen::Vector3f&n , PointT& searchPoint) {

	Eigen::Vector3f P(searchPoint.x, searchPoint.y, searchPoint.z);

	float dot_temp = P.dot(n);
	float n_St = sqrt(P.dot(P));
	float n_N = sqrt(n.dot(n));

	inc = acos(dot_temp / (n_St * n_N));
}
struct Point_IA
{
	float IA;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_IA,           // here we assume a XYZ + "test" (as fields)
	(float, IA, IA)
)

// Determination of the incidence angle for the whole point cloud --> parallel computing using omp
// cloud_result = "vector" containing all the incidence angles
// cloud = input point cloud
// kdtree = KD-Tree which cloud as input
//K = number of K neighboors for search
template <typename PointT>  
void Pcl_IA(pcl::PointCloud<Point_IA>& cloud_result, const pcl::PointCloud<PointT>& cloud,const pcl::KdTreeFLANN<pcl::PointXYZI> kdtree,  int& K) {

	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	
	int cnt = omp_get_num_procs();
	#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i < cloud.size() - 1; i++) {

		vector<int> pointIdxKNNSearch(K);
		vector<float> pointKNNSquaredDistance(K);
		pcl::PointXYZI Ptemp;
		Ptemp.x = cloud.points[i].x;
		Ptemp.y = cloud.points[i].y;
		Ptemp.z = cloud.points[i].z;


		if (isnan(Ptemp.x) == false) {

			if (kdtree.nearestKSearch(Ptemp, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {


				Eigen::Vector4f plane_parameters1;
				Eigen::Vector3f n1;
				float curvature;
				float inc;
				pcl::computePointNormal(cloud, pointIdxKNNSearch, plane_parameters1, curvature);
				n1 = plane_parameters1.head<3>();
				Incidence_angle(inc, n1, Ptemp);
				inc = abs(inc * 180 / M_PI - 90);
				
				cloud_result.points[i].IA = inc;

			}
		}
	}
}
// Spotsize
struct Point_spot
{
	float spot_ma;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_spot,           // here we assume a XYZ + "test" (as fields)
	(float, spot_ma, spot_ma)
)

// Calculating spot size
// cloud_result = "vector" cointaining all the spot sizes
// cloud = input cloud
// cloud_inc =  result "vector" containing incidence angle for each point
// d0 = spot size at the beginning [m]
// div = divergence angle [rad]
template <typename PointT>
void Spotsize(pcl::PointCloud<Point_spot>& cloud_result, const pcl::PointCloud<PointT>& cloud, pcl::PointCloud<Point_IA>& cloud_inc, const float& d0, const float& div) {

	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= cloud.size() - 1; i++) {

		if (isnan(cloud.points[i].x) != true || isnan(cloud.points[i].y) != true || isnan(cloud.points[i].z) != true) {
			float D = sqrt(pow(cloud.points[i].x, 2) + pow(cloud.points[i].y, 2) + pow(cloud.points[i].z, 2));
			float inc = cloud_inc.points[i].IA * M_PI / 180.0f;
			inc = abs(inc - M_PI / 2);
			float mA = d0 + 2. * D * (sin(2 * div)) / (cos(2. * inc) + cos(2 * div));
			cloud_result.points[i].spot_ma = mA;

		}
	}

}


// Edge points
// Deciding if point is edge or not 
// clou = input point cloud
// edge = 1 -> edge point || edge = 0 -> no edge point
// indices = indices of the k- neighboors
// threshold = lambda corresponding to Ahmed et al 
// Res = Resolution of the neighboorhood = smalles distance from point to k-neighboors

template <typename PointT> void
edge_detection(const pcl::PointCloud<PointT>& cloud, const std::vector<int>& indices, const float& threshold, const float& Res, float& edge) {

	Eigen::Vector3d centroid = Eigen::Vector3d::Zero(3);

	for (const int& index : indices)
	{
		centroid[0] += cloud[index].x;
		centroid[1] += cloud[index].y;
		centroid[2] += cloud[index].z;
	}
	centroid /= (indices.size());


	float Dist = sqrt(pow(cloud[indices[0]].x - centroid[0], 2) + pow(cloud[indices[0]].y - centroid[1], 2) + pow(cloud[indices[0]].z - centroid[2], 2));
	if (Dist > Res * threshold) {
		edge = 1;
	}
	else {
		edge = 0;
	}
	
}


struct Point_edge
{
	float edge;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_edge,           // here we assume a XYZ + "test" (as fields)
	(float, edge, edge)
)

// Determination of edge points for whole point cloud
// cloud_result = "vector" containing information if point is edge (1) or not (0)
// cloud = input point cloud
// kdtree = Kdtree with cloud as input
// threshold = lambda
// K = K-Neighbourhood
template <typename PointT> inline void
edge_detection_calculation(pcl::PointCloud<Point_edge>& cloud_result, const pcl::PointCloud<PointT>& cloud, const pcl::KdTreeFLANN<pcl::PointXYZI> kdtree,const float& threshold, int& K) {


	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= cloud.size() - 1; i++) {

		vector<int> pointIdxKNNSearch(K);
		vector<float> pointKNNSquaredDistance(K);
		float edge;
		pcl::PointXYZI Ptemp;
		if (isnan(cloud.points[i].x) != true || isnan(cloud.points[i].y) != true || isnan(cloud.points[i].z) != true || isnan(cloud.points[i].intensity) || cloud.points[i].x == 0 && cloud.points[i].y == 0 && cloud.points[i].z == 0) {
			Ptemp.x = cloud.points[i].x;
			Ptemp.y = cloud.points[i].y;
			Ptemp.z = cloud.points[i].z;

			if (kdtree.nearestKSearch(Ptemp, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
				edge_detection(cloud, pointIdxKNNSearch, threshold, sqrt(pointKNNSquaredDistance[1]), edge);
				edge_detection(cloud, pointIdxKNNSearch, threshold, sqrt(pointKNNSquaredDistance[1]), edge);
				cloud_result.points[i].edge = edge;
			}
			else {
				cloud_result.points[i].edge = NAN;
			}

		}
	}
	
}


// Filterung


template <typename PointT> inline void
punkt_filterung(pcl::PointIndices& Indices, const pcl::PointCloud<PointT>& cloud, const float& t_angle, const float& t_spot, const float& t_std_lokal, const bool& edge, const float& min_intens, const float& max_intens) {
	
	//ofstream myfile;
	//myfile.open("example.txt");
	for (int i = 0; i <= cloud.size() - 1; i++) {
		bool dec = false;
		//myfile << min_intens << " " << cloud.points[i].spot_ma << endl;
		//if (cloud.points[i].intensity < min_intens) {
		//	//myfile << min_intens << " " << cloud.points[i].intensity << endl;
		//	dec = true;
		//}
		if (cloud.points[i].std_lokal > t_std_lokal || cloud.points[i].IA < t_angle || cloud.points[i].spot_ma > t_spot || cloud.points[i].intensity < min_intens || cloud.points[i].intensity > max_intens)
		{
			dec = true;

			
		}
		if (edge == true) {
			if (cloud.points[i].edge == 1) {
				dec = true;
			}
		}
		if (dec == true) {
			Indices.indices.push_back(i);
		
		}

	}
};

// Auflösung
struct Point_reso
{
	float resolution;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_reso,           // here we assume a XYZ + "test" (as fields)
	(float, resolution, resolution)
)

template <typename PointT> inline void
resolution_calculation(pcl::PointCloud<Point_reso>& cloud_result, const pcl::PointCloud<PointT>& cloud, const float& d0, const float& div,const float& scan_resolution, int decision) {

	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= cloud.size() - 1; i++) {
		if (isnan(cloud.points[i].x) != true || isnan(cloud.points[i].y) != true || isnan(cloud.points[i].z) != true || isnan(cloud.points[i].intensity) || cloud.points[i].x == 0 && cloud.points[i].y == 0 && cloud.points[i].z == 0) {
			float res = 0;
			// Strecke
			if (decision == 1) {
				float dist = sqrt(pow(cloud.points[i].x, 2) + pow(cloud.points[i].y, 2) + pow(cloud.points[i].z, 2));
				res = d0 + (dist)*div;
			
			}

			// Winkel
			if (decision == 2) {
				float dist = sqrt(pow(cloud.points[i].x, 2) + pow(cloud.points[i].y, 2) + pow(cloud.points[i].z, 2));
				res = dist * scan_resolution / (180 / M_PI);
			
			}

			// kombiniert
			if (decision == 3) {
				float dist = sqrt(pow(cloud.points[i].x, 2) + pow(cloud.points[i].y, 2) + pow(cloud.points[i].z, 2));
			float res1 = d0 + (dist)*div;
			float res2  = dist * scan_resolution / (180 / M_PI);
			res = max(res1, res2);
			}
			cloud_result.points[i].resolution = res;

		}

	}
}

////
template <typename PointT> inline void
resolution_calculation_with_spot(pcl::PointCloud<Point_reso>& cloud_result, const pcl::PointCloud<PointT>& cloud, const pcl::PointCloud<Point_spot>& cloud_spot,const float& d0, const float& div, const float& scan_resolution) {

	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());
	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= cloud.size() - 1; i++) {
		if (isnan(cloud.points[i].x) != true || isnan(cloud.points[i].y) != true || isnan(cloud.points[i].z) != true || isnan(cloud.points[i].intensity) || cloud.points[i].x == 0 && cloud.points[i].y == 0 && cloud.points[i].z == 0) {
			float res = 0;
			// Strecke
		
			// kombiniert
			
				float dist = sqrt(pow(cloud.points[i].x, 2) + pow(cloud.points[i].y, 2) + pow(cloud.points[i].z, 2));
				float res1 = d0 + (dist)*div;
				float res2 = dist * scan_resolution / (180 / M_PI);
				//res = max();
				
				res = std::max({ res1, res2,cloud_spot[i].spot_ma });
			cloud_result.points[i].resolution = res;

		}

	}
}
////
template <typename PointT> inline void
resolution_Filter(pcl::PointCloud<PointT>& cloud, pcl::KdTreeFLANN<pcl::PointXYZI> kdtree) {

	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i < cloud.size() - 1; i++) {

		vector<int> pointIdxKNNSearch(2);
		vector<float> pointKNNSquaredDistance(2);
		pcl::PointXYZI Ptemp;
		Ptemp.x = cloud.points[i].x;
		Ptemp.y = cloud.points[i].y;
		Ptemp.z = cloud.points[i].z;
		if (kdtree.nearestKSearch(Ptemp, 2, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
			float dist = sqrt(pointKNNSquaredDistance[1]);
		/*	if (cloud.points[i].resolution < dist) {
				Indices.indices.push_back(i);
			
			}*/
			cloud.points[i].resolution = cloud.points[i].resolution - dist;
		}
	}
}

/// <summary>
/// 
/// </summary>

struct Point_curvature
{
	float curvature;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_curvature,           // here we assume a XYZ + "test" (as fields)
	(float, curvature, curvature)
)

template <typename PointT>
void Pcl_curvature(pcl::PointCloud<Point_curvature>& cloud_result, const pcl::PointCloud<PointT>& cloud, const pcl::KdTreeFLANN<pcl::PointXYZI> kdtree, int& K) {

	cloud_result.width = cloud.width;
	cloud_result.height = cloud.height;
	cloud_result.resize(cloud.size());

	int cnt = omp_get_num_procs();
#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i < cloud.size() - 1; i++) {

		vector<int> pointIdxKNNSearch(K);
		vector<float> pointKNNSquaredDistance(K);
		pcl::PointXYZI Ptemp;
		Ptemp.x = cloud.points[i].x;
		Ptemp.y = cloud.points[i].y;
		Ptemp.z = cloud.points[i].z;
		if (kdtree.nearestKSearch(Ptemp, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
			Eigen::Vector4f plane_parameters1;
			Eigen::Vector3f n1;
			float curvature;
			pcl::computePointNormal(cloud, pointIdxKNNSearch, plane_parameters1, curvature);
			n1 = plane_parameters1.head<3>();
			cloud_result.points[i].curvature = curvature;
		}
	}
}


void transformation_matrix(Eigen::Matrix4f& mat4, Eigen::Quaternionf& Quater_rot, Eigen::Vector4f& Pose4d) {

	Eigen::Matrix3f mat3 = Quater_rot.toRotationMatrix();
	mat4.block(0, 0, 3, 3) = mat3;
	mat4.col(3) = Pose4d;
}




// Voxelfilter
//template <typename PointT>
//void create_voxel(pcl::PCLPointCloud2& cloud_result, const pcl::PointCloud<PointT>& cloud) {
//	
//	pcl::PCLPointCloud2::Ptr cloud_temp(new pcl::PCLPointCloud2());
//	pcl::toPCLPointCloud2(cloud, *cloud_temp);
//	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
//	sor.setInputCloud(cloud_temp);
//	sor.setLeafSize(0.1f, 0.1f, 0.1f);
//	sor.filter(cloud_result);
//
//}





///////////////////////////////// Authentizität /////////////////////////
struct Point_authent
{
	float auth_gesamt;
	PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(Point_authent,           // here we assume a XYZ + "test" (as fields)
	(float, auth_gesamt, auth_gesamt)
)



