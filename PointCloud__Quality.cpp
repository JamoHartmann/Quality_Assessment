// PointCloud__Quality.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS
#include "Modules/IO/IO_E57.h"
#include <iostream>
#include <omp.h>
#include <pcl/console/parse.h>
#include "Modules/Quality/QualityAssessment.h"
#include <ctime>
#include <pcl/common/transforms.h>
struct Laserscanner
{
	float a;
	float b;
	float c;
	float sigma_angle;
	float d0;
	float div;
	float aufl;
	float linear_error;
};

struct Viewpoint
{
	Eigen::Vector3f Pose;
	Eigen::Quaternionf Quater_rot;
	Eigen::MatrixXf SIGMA_LL_Registrierung = Eigen::MatrixXf::Zero(7, 7);;

};

int main(int argc, const char* argv[])
{
    
	bool local = false;
	bool global = false;
	bool Incidence = false;
	bool Edge = false;
	std::string resu = "result.ply";
	e57::ustring filename;
	Laserscanner LS;
	Viewpoint VP;
	int K_IA, K_Edge;
	float lambda_Edge;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);





	// Name of the point cloud 
	int P = (pcl::console::find_argument(argc, argv, "-P"));
	if (P != -1) {
		filename = argv[P + 1];
		//filename.
		std::cout << "path of point cloud: " << filename << std::endl;
	}
	else {
		std::cout << "\033[1;31m point cloud not found\033[0m\n";

		return 0;
	}

	// Name of the saved point cloud
	int R = (pcl::console::find_argument(argc, argv, "-R"));
	if (R != -1) {
		resu = argv[R + 1];

		std::cout << "path of saved point cloud: " << resu << std::endl;
	}
	else {
	
		std::cout << "\033[1;31m Point cloud will not be saved\033[0m\n";
	}

	/// Reading sensor information:
	const char* c1;
	int S = (pcl::console::find_argument(argc, argv, "-S"));
	if (S != -1) {
		c1 = argv[S + 1];
		std::ifstream infile;
		infile.open(c1);
		std::string line;
		std::getline(infile, line);
		infile >> LS.a >> LS.b >> LS.c >> LS.sigma_angle >> LS.linear_error >> LS.d0 >> LS.div >> LS.aufl;
		std::cout << "A=" << LS.a << " B=" << LS.b << " C=" << LS.c << " Sigma_Winkel=" << LS.sigma_angle << "Linear Error= " << LS.linear_error<< " Spotgroeße=" << LS.d0 << " Divergenzwinkel= " << LS.div << " Auflösung= " << LS.div << std::endl;
		std::cout << "******************************************" << std::endl;
		local = true;
	}
	else {
		std::cout << "\033[1;31m Sensor information could not be read\033[0m\n";
	}
	

	const char* c2;
	int VKV = (pcl::console::find_argument(argc, argv, "-VKV"));
	if (VKV != -1) {
		c2 = argv[VKV + 1];
		std::ifstream infile;
		infile.open(c2);
		float cov;
		std::string line;
		std::getline(infile, line);
		int j = 0;
		int k = 0;
		for (int i = 0; i <= 27; i++) {
			infile >> cov;
			VP.SIGMA_LL_Registrierung(k, j) = cov;
			VP.SIGMA_LL_Registrierung(j, k) = cov;

			k = k + 1;
			if (k == 7) {
				j = j + 1;
				k = j;
			}
		}
		infile.close();
		std::cout << VP.SIGMA_LL_Registrierung << std::endl;
		global = true;
	}
	else {
		std::cout << "\033[1;31m VCV not available -> Use identity matrix\033[0m\n";
	}


	int IA = (pcl::console::find_argument(argc, argv, "-IA"));
	if (IA != -1) {
		K_IA = strtol(argv[IA + 1], NULL, 10);
		//filename.
		std::cout << "K-Neighborhood for Incidence angle:" << K_IA << std::endl;
		Incidence = true;
	}
	else {
		std::cout << "\033[1;31m Incidence angle will not be calculated\033[0m\n";
		Incidence = false;
	
	}

	int E = (pcl::console::find_argument(argc, argv, "-Edge"));
	if (E != -1) {
		K_Edge = strtol(argv[E +1], NULL, 10);
		//filename.
		std::cout << "K-Neighborhood for Edge:" << K_Edge << std::endl;
		Edge = true;
	}
	else {
		std::cout << "\033[1;31m Edge will not be calculated\033[0m\n";
		Edge = false;
	}


	///////////////////////////////////////////////////////////////////////////
	// 
	// CALCULATION OF THE FEATURES ///


	std::time_t time_now = time(0);
	tm* ltm = localtime(&time_now);



	std::cout << "-----------------------------------------------------" << std::endl;
	std::cout << "\033[1;32m Start of calculation \033[0m" << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";

	// Loading e57 point cloud

	time_now = time(0);
	ltm = localtime(&time_now);
	std::cout << "-----------------------------------------------------" << std::endl;
	std::cout << "\033[1;32m Loading Pointcloud \033[0m" << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";
	bool rgb, intensity, xyz, org;
	////long test = filename.c_str();
	//e57::ustring test= filename.c_str();
	//cout << filename2.size() << endl;
	
	Pcl_type(rgb, intensity, xyz, org, filename);
	
	if (intensity == false)
	{
		std::cout << "\033[1;31m No intensity values\033[0m\n";
		return 0;
	}
	if (xyz == false) {
		std::cout << "\033[1;31m No point values\033[0m\n";
		return 0;
	}

	if (org == false) {
		std::cout << "\033[point cloud is not organized\033[0m\n";
	
	} 


	read_e57_XYZI(*cloud, filename, org);

	std::cout << "Point cloud information:\n" << "Width = " << cloud->width << "\n Height = " << cloud->height << "\n Num. Points = " << cloud->size() << std::endl;
	time_now = time(0);
	ltm = localtime(&time_now);
	std::cout << "\033[1;32m Loaded Pointcloud \033[0m" << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";

	//////////////// Local precision 

	std::cout << "-----------------------------------------------------" << std::endl;
	
	pcl::PointCloud<Point_3d_lokal>::Ptr cloud_local(new pcl::PointCloud<Point_3d_lokal>);
	if (local == true) {
		std::cout << "\033[1;32m local precision \033[0m";
		calculate_3D_std_lokal(*cloud_local, *cloud, LS.a, LS.b, LS.c, LS.sigma_angle, LS.sigma_angle, LS.linear_error);
		time_now = time(0);
		ltm = localtime(&time_now);
		std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";
	}

	//////////////// Global precision 

	pcl::PointCloud<Point_3d_global>::Ptr cloud_global(new pcl::PointCloud<Point_3d_global>);
	
	// Sensor orientation saved in e57 file is assumed to be the used transformation
	if (global == true) {

		bool glob = true;
		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << "\033[1;32m global precision \033[0m";

		Pcl_std_3D_global(*cloud_global, *cloud, glob, cloud->sensor_orientation_, VP.SIGMA_LL_Registrierung, LS.sigma_angle, LS.sigma_angle, LS.a, LS.b, LS.c, LS.linear_error);

		time_now = time(0);
		ltm = localtime(&time_now);
		std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";
	}
	
	//////////////// Incidence angle  + Spotsize
	pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointCloud<Point_IA>::Ptr cloud_IA(new pcl::PointCloud<Point_IA>);
	pcl::PointCloud<Point_spot>::Ptr cloud_spot(new pcl::PointCloud<Point_spot>);
	if (Incidence == true) {
		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << "\033[1;32m Incidence angle \033[0m";
		


		
		
		Pcl_IA(*cloud_IA, *cloud, kdtree, K_IA);

		time_now = time(0);
		ltm = localtime(&time_now);
		std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";





		
		std::cout << "\033[1;32m Spot size \033[0m";
		Spotsize(*cloud_spot, *cloud, *cloud_IA, LS.d0, LS.aufl);
		time_now = time(0);
		ltm = localtime(&time_now);
		std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";
	}
	//////////// Edge 
	pcl::PointCloud<Point_edge>::Ptr cloud_edge(new pcl::PointCloud<Point_edge>);
	if (Edge == true) {
		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << "\033[1;32m Edge points \033[0m";

		//pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
		//kdtree.setInputCloud(cloud);
		
		float threshold = 1;
		
		edge_detection_calculation(*cloud_edge, *cloud, kdtree, threshold, K_Edge);

		time_now = time(0);
		ltm = localtime(&time_now);
		std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";

	}

	////////// Save PCL as ply
	Eigen::Matrix4f mat4;
	transformation_matrix(mat4, cloud->sensor_orientation_, cloud->sensor_origin_);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::transformPointCloud(*cloud, *cloud_tr, mat4, true);

	pcl::PCLPointCloud2::Ptr firstCloud(new pcl::PCLPointCloud2());
	pcl::toPCLPointCloud2(*cloud_tr, *firstCloud);
	pcl::PCLPointCloud2::Ptr cloud_result(new pcl::PCLPointCloud2());
	// Local
	if (cloud_local->size() > 10) {
		pcl::PCLPointCloud2::Ptr tempCloud(new pcl::PCLPointCloud2());
		pcl::PCLPointCloud2::Ptr tempCloud2(new pcl::PCLPointCloud2());
		

		pcl::toPCLPointCloud2(*cloud_local, *tempCloud);
		pcl::concatenateFields(*firstCloud, *tempCloud, *tempCloud2);
		firstCloud = tempCloud2;
	}
	// Global
	//tempCloud2 = cloud_result;


	if (cloud_global->size() > 10) {
		pcl::PCLPointCloud2::Ptr tempCloud(new pcl::PCLPointCloud2());
		pcl::PCLPointCloud2::Ptr tempCloud2(new pcl::PCLPointCloud2());

		pcl::toPCLPointCloud2(*cloud_global, *tempCloud);
		pcl::concatenateFields(*firstCloud, *tempCloud, *tempCloud2);
		firstCloud = tempCloud2;

	}

	if (cloud_IA->size() > 10) {
		pcl::PCLPointCloud2::Ptr tempCloud(new pcl::PCLPointCloud2());
		pcl::PCLPointCloud2::Ptr tempCloud2(new pcl::PCLPointCloud2());

		pcl::toPCLPointCloud2(*cloud_IA, *tempCloud);
		pcl::concatenateFields(*firstCloud, *tempCloud, *tempCloud2);
		firstCloud = tempCloud2;

	}

	if (cloud_spot->size() > 10) {
		pcl::PCLPointCloud2::Ptr tempCloud(new pcl::PCLPointCloud2());
		pcl::PCLPointCloud2::Ptr tempCloud2(new pcl::PCLPointCloud2());

		pcl::toPCLPointCloud2(*cloud_spot, *tempCloud);
		pcl::concatenateFields(*firstCloud, *tempCloud, *tempCloud2);
		firstCloud = tempCloud2;

	}

	if (cloud_edge->size() > 10) {
		pcl::PCLPointCloud2::Ptr tempCloud(new pcl::PCLPointCloud2());
		pcl::PCLPointCloud2::Ptr tempCloud2(new pcl::PCLPointCloud2());

		pcl::toPCLPointCloud2(*cloud_edge, *tempCloud);
		pcl::concatenateFields(*firstCloud, *tempCloud, *tempCloud2);
		firstCloud = tempCloud2;

	}

	std::cout << "\033[1;32m Saving \033[0m";
	save_pointcloud2_ply(*firstCloud, resu, cloud->sensor_orientation_, cloud->sensor_origin_);
	time_now = time(0);
	ltm = localtime(&time_now);
	std::cout << "[" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "]\n";
	1 + 1;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
