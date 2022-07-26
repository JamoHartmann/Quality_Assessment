#pragma once
#include <math.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/range_image/range_image.h>
//#include <pcl/visualization/image_viewer.h>
//#include <pcl/visualization/common/float_image_utils.h> 
//#include <pcl/io/png_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/crop_box.h>
#include "CImg.h"
#include <vtkCubeSource.h>
#include <vtkCleanPolyData.h>
#include <vtkGlyph3DMapper.h>
//#include <vtkGPUVolumeRayCastMapper.h>
//#include <vtkNamedColors.h>
#include <vtkStructuredGrid.h>
#include <omp.h>
#include <pcl/surface/concave_hull.h>
using namespace cimg_library;
template <typename PointT>
void create_RangeImage_from_pcl(pcl::RangeImage& RangeImage, const float& angular_resolution, pcl::PointCloud<PointT>& cloud) {

	float angularResolution = (float)(angular_resolution * (M_PI / 180.0f));  //   1.0 degree in radians
	float maxAngleWidth = (float)(360.0f * (M_PI / 180.0f));  // 360.0 degree in radians
	float maxAngleHeight = (float)(180.0f * (M_PI / 180.0f));  // 180.0 degree in radians
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	float noiseLevel = 0.0;
	float minRange = 0.0f;
	int borderSize = 1;

	Eigen::Matrix4f mat4;
	Eigen::Matrix3f mat3 = cloud.sensor_orientation_.toRotationMatrix();
	mat4.block(0, 0, 3, 3) = mat3;
	mat4.col(3) = cloud.sensor_origin_;

	Eigen::Affine3f Pose;
	Pose.matrix() = mat4;
	RangeImage.createFromPointCloud(cloud, angularResolution, maxAngleWidth, maxAngleHeight,
		Pose, coordinate_frame, noiseLevel, minRange, borderSize);


}


//void save_rangeImage_png(pcl::RangeImage& RangeImage, std::string& file) {
//	float* ranges = RangeImage.getRangesArray();
//	unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges, RangeImage.width, RangeImage.height);
//	pcl::io::saveRgbPNGFile(file, rgb_image, RangeImage.width, RangeImage.height);
//
//}

template <typename PointT>
void getMinMax_from_Cloud_list(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& list_cloud, PointT &minP_end, PointT& maxP_end) {

	pcl::PointCloud<pcl::PointXYZI>::Ptr temp(new pcl::PointCloud<pcl::PointXYZI>);


	// Erst müssen die Grenzen Bestimmt werden:
	for (int i = 0; i <= list_cloud.size() - 1; i++) {
		pcl::PointXYZI minP, maxP;
		pcl::getMinMax3D(*list_cloud[i], minP, maxP);
		temp->push_back(minP);
		temp->push_back(maxP);
	}

	
	pcl::getMinMax3D(*temp, minP_end, maxP_end);
}

template <typename PointT>
void calculate_3D_Occupacy(std::vector<pcl::RangeImage::Ptr>& list_rangeImage, const PointT& minPt, const PointT& maxPt, float step, CImg<int>& Result_Image, CImg<int>& Img_pcl) {
	
	int xd = ceil((maxPt.x - minPt.x) / step);
	int yd = ceil((maxPt.y - minPt.y) / step);
	int zd = ceil((maxPt.z - minPt.z) / step);

	CImg<int> visu3(xd, yd, zd, 1, 0);
	CImg<int> visu3_2(xd, yd, zd, 1, 0);
	int cnt = omp_get_num_threads();
	#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= xd - 1; i++) {
		for (int j = 0; j <= yd - 1; j++) {

			
			for (int k = 0; k <= zd - 2; k++) {
				Eigen::Vector3f pp(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step); // Punkt der überprüft wird
				int sum_vis = 0;
				int sum_vis_2 = 0;
				int inf_t = 0; // Zählen von Inf Range -> keine Beobachtungen -> Bei mher als 2 die Inf sind wird der Punkt als sichtbar deklariert (Annahme Himmel)
				for (int l = 0; l <= list_rangeImage.size() - 1; l++) {
					float range_diff = list_rangeImage[l]->getRangeDifference(pp);
					// Wenn Inf dann keine Beoabchtung -> Datenlücke oder Himmel -> Deswegen, wenn in mehreren Punktwolken nan = Himmel
					if (std::isinf(range_diff) == true) {
						inf_t = inf_t + 1;
					}
					// Range Difference größer null = Sichtbar
					if (range_diff > 0) {
						sum_vis = 1;
					}
					if (abs(range_diff) <= step) {
						sum_vis_2 = 1;
					}
				}
				// Wenn sum_vis == 0 gibt es keinen sichtbaren Punkt Deshalb als verdeckt markieren:
				if (sum_vis == 0 && inf_t <= 2) {
					visu3(i, j, k) = 1;
					//visu3_2(i, j, k) = 1;
				}
				else {
					visu3(i, j, k) = 0;
					//visu3_2(i, j, k) = 0;
				}
				if (sum_vis_2 == 1) {
					visu3_2(i, j, k) = 1;
				}
				else {
					visu3_2(i, j, k) = 0;
				}

			}
		}
	}
	Result_Image = visu3;
	Img_pcl = visu3_2;
}

void calc_occ_map(CImg<float>& Result_Image,CImg<float>& Colorbar,float& sichtbar, float& verdeckt, CImg<int>Occ_3d, const int zmin, const int zmax) {

	CImg<float> Temp = Occ_3d.get_slice(zmin);
	for (int i = zmin + 1; i <= zmax;i++) {
		Temp.operator+=(Occ_3d.get_slice(i));
	
	}
	verdeckt= Temp.sum() / (Temp.height() * Temp.width() * (zmax + 1)) * 100;
	sichtbar = 100 - verdeckt;

	// Farbgebung
	const unsigned char	black[] = { 0,0,0 };
	CImg <> colormap = CImg<>::jet_LUT256();
	
	colormap.resize(40, 200);
	colormap = colormap.mirror('y');
	colormap.draw_axes(0, 0, zmax, zmin, black, 1.0f, 2, 5, 1.0f, 0.0f);
	Temp.operator/=(zmax - zmin+1);
	Temp.operator*=(255);

	Temp.map(CImg<>::jet_LUT256());

	Result_Image = Temp;
	Colorbar = colormap;
}

template <typename PointT>
void Draw_heigth(CImg<float>& Height_Image, const int zmin, const int zmax, const int z_all, const PointT& minPt, const PointT& maxPt) {
	const unsigned char	black[] = { 0,0,0 }, white[] = { 255,255,255 }, red[] = { 255 , 0, 0};

	CImg<float> Hei(80, z_all, 1, 1, 0);
	//Hei.draw_line(0, z_all - zmin, 40, z_all - zmin, white);
	Hei.draw_line(0, z_all- zmax, 20, z_all - zmax, red);
	Hei.draw_line(0, z_all - zmin, 20, z_all - zmin, white);
	Hei.resize(50, 500);
	Hei.draw_axes(0, 0, maxPt.z, minPt.z, white, 1.0f, 2, 5, 1.0f, 0.0f);

	Height_Image = Hei;
}


template <typename PointT>
void calculate_occupied_voxel_Pcl(vtkNew<vtkPoints>& points, CImg<int>& Result,const int zmin,const int zmax, const PointT& minPt,const float& step) {


	for (int i = 0; i <= Result.width(); i++) {

		for (int j = 0; j <= Result.height(); j++) {
			for (int k = zmin; k <= zmax; k++) {

				if (Result(i, j, k) == 1) {
					if (i > 0 && j > 0 && k > zmin && i < Result.width() && j < Result.height() && k < zmax) {
						if (Result(i - 1, j, k) == 1 && Result(i + 1, j, k) == 1 && Result(i, j - 1, k) == 1 && Result(i, j + 1, k) == 1 && Result(i, j, k - 1) == 1 && Result(i, j, k + 1) == 1) {

						}
						else {
							points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
						}
					}
					else {
						points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
					}
				}

			}

		}
	}



}

template <typename PointT>
void calculate_visible_voxel_Pcl(vtkNew<vtkPoints>& points, vtkFloatArray& scalars, CImg<int>& Result, const int zmin, const int zmax, const PointT& minPt, const float& step) {


	for (int i = 0; i <= Result.width(); i++) {

		for (int j = 0; j <= Result.height(); j++) {
			for (int k = zmin; k <= zmax; k++) {

				if (Result(i, j, k) == 1) {
					points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
					scalars.InsertNextValue(k);
				}

			}

		}
	}



}





/////////
template <typename PointT>
void calculate_occupied_voxel_Pcl_polygon(vtkNew<vtkPoints>& points, CImg<int>& Result,CImg<int>Polygon, const int xmin, const int xmax, const int ymin, const int ymax,const int zmin, const int zmax, const PointT& minPt, const float& step) {


	for (int i = xmin; i <= xmax; i++) {

		for (int j = ymin; j <= ymax; j++) {
			for (int k = zmin; k <= zmax; k++) {

				if (Result(i, j, k) == 1 && Polygon(i, j) == 1) {
					if (i >=xmin && j >= ymin && k > zmin && i <= xmax && j <= ymax && k <= zmax &&i > 0 && j > 0  && i < Result.width() && j < Result.height() ) {
						if (Result(i - 1, j, k) == 1 && Result(i + 1, j, k) == 1 && Result(i, j - 1, k) == 1 && Result(i, j + 1, k) == 1 && Result(i, j, k - 1) == 1 && Result(i, j, k + 1) == 1) {

						}
						else {
							points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
						}
					}
					else {
						points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
					}
				}

			}

		}
	}



}

////////

void calc_visible_polygon(CImg<int>Occ_3d, const int zmin, const int zmax,CImg<int>& Polygon,float& sichtbar, float& verdeckt) {

	CImg <int> TT = Occ_3d;
	for (int i = 0; i < TT.width(); i++) {
	
		for (int j = 0; j < TT.height(); j++) {
			for (int k = 0; k < TT.depth(); k++) {
			
				if (Polygon(i, j, 0) == 0) {
				
					TT(i, j, k) = 0;
				}
			
			}
		}
	}


	CImg<float> Temp =TT.get_slice(zmin);
	for (int i = zmin + 1; i <= zmax; i++) {
		Temp.operator+=(TT.get_slice(i));
	}
	verdeckt = Temp.sum() / (Temp.height() * Temp.width() * (zmax + 1)) * 100;
	sichtbar = 100 - verdeckt;


}
///////


void combine_pcl(std::vector <pcl::PointCloud<pcl::PointXYZI>::Ptr>& list_cloud, pcl::PointCloud<pcl::PointXYZI>& cloud_result) {

	for (int i = 0; i < list_cloud.size(); i++) {
	
		cloud_result += *list_cloud[i];
	}

}

void uniform_sampling(const pcl::PointCloud<pcl::PointXYZI>& cloud, pcl::PointCloud<pcl::PointXYZI>& cloud_result,const float minZ, const float maxZ) {
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_t(new pcl::PointCloud<pcl::PointXYZI>);
	*cloud_t = cloud;
	pcl::UniformSampling<pcl::PointXYZI> unisamp;
	unisamp.setInputCloud(cloud_t);
	unisamp.setRadiusSearch(0.2);
	unisamp.filter(*cloud_t);

	pcl::CropBox<pcl::PointXYZI> boxFilter;
	boxFilter.setMin(Eigen::Vector4f(-100000, -100000, minZ, 1.0));
	boxFilter.setMax(Eigen::Vector4f(100000, 100000, maxZ, 1.0));
	boxFilter.setInputCloud(cloud_t);
	boxFilter.filter(cloud_result);
}




////////////////////////

template <typename PointT>
void calculate_Occlusions(pcl::RangeImage& RI, const PointT& minPt, const PointT& maxPt, float step, CImg<int>& Result_Image) {

	int xd = ceil((maxPt.x - minPt.x) / step);
	int yd = ceil((maxPt.y - minPt.y) / step);
	int zd = ceil((maxPt.z - minPt.z) / step);

	CImg<int> visu3(xd, yd, zd, 1, 0);
	CImg<int> visu3_2(xd, yd, zd, 1, 0);
	int cnt = omp_get_num_threads();
	#pragma omp parallel for num_threads(cnt)
	for (int i = 0; i <= xd - 1; i++) {
		for (int j = 0; j <= yd - 1; j++) {


			for (int k = 0; k <= zd - 2; k++) {
				Eigen::Vector3f pp(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step); // Punkt der überprüft wird
				int sum_vis = 0;
				int sum_vis_2 = 0;
				int inf_t = 0; // Zählen von Inf Range -> keine Beobachtungen -> Bei mher als 2 die Inf sind wird der Punkt als sichtbar deklariert (Annahme Himmel)
				
					float range_diff = RI.getRangeDifference(pp);
					// Wenn Inf dann keine Beoabchtung -> Datenlücke oder Himmel -> Deswegen, wenn in mehreren Punktwolken nan = Himmel
					if (std::isinf(range_diff) == true) {
						inf_t = inf_t + 1;
					}
					// Range Difference größer null = Sichtbar
					if (range_diff > 0) {
						sum_vis = 1;
					}
					if (abs(range_diff) <= step) {
						sum_vis_2 = 1;
					}
				
				// Wenn sum_vis == 0 gibt es keinen sichtbaren Punkt Deshalb als verdeckt markieren:
				if (sum_vis == 0 && inf_t <= 1) {
					visu3(i, j, k) = 1;
					//visu3_2(i, j, k) = 1;
				}
				else {
					visu3(i, j, k) = 0;
					//visu3_2(i, j, k) = 0;
				}


			}
		}
	}
	Result_Image = visu3;
}
///////



template <typename PointT>
void calculate_occupied_voxel_Pcl_pol(vtkNew<vtkPoints>& points, CImg<int>& Result, CImg<int>& Polygon, const int zmin, const int zmax, const PointT& minPt, const float& step) {


	for (int i = 0; i <= Result.width(); i++) {

		for (int j = 0; j <= Result.height(); j++) {
			for (int k = zmin; k <= zmax; k++) {

				if (Result(i, j, k) == 1 && Polygon(i, j, 0) == 1) {
					if (i > 0 && j > 0 && k > zmin && i < Result.width() && j < Result.height() && k < zmax && i > 0 && j > 0 && i < Result.width() && j < Result.height()) {
						if (Result(i - 1, j, k) == 1 && Result(i + 1, j, k) == 1 && Result(i, j - 1, k) == 1 && Result(i, j + 1, k) == 1&& Result(i, j , k-1) == 1 && Result(i, j, k +1) ==1 && Polygon(i - 1, j) == 1 && Polygon(i + 1, j) == 1 && Polygon(i, j - 1) == 1 && Polygon(i, j + 1) == 1) {

						}
						else {
							points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
						//	points->InsertNextPoint(minPt.x + i * step - step/2, minPt.y + j * step - step / 2, minPt.z + k * step - step / 2);
						}
					}
					else {
						points->InsertNextPoint(minPt.x + i * step, minPt.y + j * step, minPt.z + k * step);
						//points->InsertNextPoint(minPt.x + i * step - step / 2, minPt.y + j * step - step / 2, minPt.z - k * step + step / 2);
					}
				}

			}

		}
	}



}


/// <summary>
/// ///////////////////
/// </summary>
/// <param name="cloud"></param>
void Calc_unvisible_cells(pcl::PointCloud<pcl::PointXYZI>& cloud) {
	pcl::PointXYZI minP, maxP;
	pcl::PointXYZ pt;
	pcl::RangeImage RI;
	create_RangeImage_from_pcl(RI, 1, cloud);
	/*std::string range_file = "Color/RangeImage.png";
	save_rangeImage_png(RI, range_file);*/


	// Verdeckungen

	pcl::getMinMax3D(cloud, minP, maxP);

	float step = 0.2;
	int xd = ceil((maxP.x - minP.x) / step);
	int yd = ceil((maxP.y - minP.y) / step);
	int zd = ceil((maxP.z - minP.z) / step);

	CImg<int> visu3(xd, yd, zd, 1, 0);

	for (int i = 1; i < cloud.size(); i++) {
		int xx = (cloud.points[i].x - minP.x) / step;
		int yy = (cloud.points[i].y - minP.y) / step;
		int zz = (cloud.points[i].z - minP.z) / step;
		if (xx >= 0 && yy >= 0 && zz >= 0) {
			visu3(xx, yy, zz) = 1;//%visu3(xx, yy) + 1;
		}
	}

	CImg<float> Temp = visu3.get_slice(0);
	////
	for (int i = 1; i <= zd - 1; i++) {
		Temp.operator+=(visu3.get_slice(i));
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cl1(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i <= Temp.width() - 1; i++) {

		for (int j = 0; j <= Temp.height() - 1; j++) {
			if (Temp(i, j, 0) > 3) {
				pt.x = i;
				pt.y = j;
				pt.z = 0;
				cl1->push_back(pt);
			}
		}
	}


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ConcaveHull<pcl::PointXYZ> chull;
	chull.setInputCloud(cl1);
	chull.setAlpha(100);
	chull.reconstruct(*cloud_hull);

	CImg<int> poly(cloud_hull->size(), 2);
	//std::vector<int> xm1;
	//std::vector<int> ym1;
	for (int i = 0; i < cloud_hull->size(); i++) {
		poly(i, 0) = cloud_hull->points[i].x;
		poly(i, 1) = cloud_hull->points[i].y;
	}
	int red[] = { 1 };

	CImg<int> Polygon_t;


	Polygon_t = CImg<int>(Temp.width(), Temp.height(), 1, 1, 0);

	Polygon_t.draw_polygon(poly, red);
	//Polygon_t.map(CImg<>::jet_LUT256());
	//Polygon_t.save_bmp("Color/Polygon.bmp");
	//	Polygon_t.display();
	CImg<int> Result_Image;
	calculate_Occlusions(RI, minP, maxP, step, Result_Image);
	float sichtbar;
	float verdeckt;
	int zmin = 0;
	int zmax = Result_Image.depth();

	CImg<float> Colorbar;
	CImg<float> Occ_map;
	calc_occ_map(Occ_map, Colorbar, sichtbar, verdeckt, Result_Image, zmin, zmax);
	for (int i = 0; i < Polygon_t.width(); i++) {

		for (int j = 0; j < Polygon_t.height(); j++) {


			if (Polygon_t(i, j) == 0) {
				Occ_map(i, j, 0) = -1;
				Occ_map(i, j, 1) = -1;
				Occ_map(i, j, 2) = -1;
			}
		}
	}
	//Occ_map.map(CImg<>::jet_LUT256());
	Occ_map.save_bmp("Color/Occlusion.bmp");

	//////
	vtkNew<vtkCubeSource> cubeSource;
	cubeSource->SetXLength(step);
	cubeSource->SetYLength(step);
	cubeSource->SetZLength(step);

	vtkNew<vtkPoints> points;
	zmax = Result_Image.depth()-2;
	calculate_occupied_voxel_Pcl_pol(points, Result_Image, Polygon_t, zmin, zmax, minP, step);
	vtkNew<vtkPolyData> polydata;
	polydata->SetPoints(points);
	vtkNew<vtkGlyph3DMapper> glyph3Dmapper;
	glyph3Dmapper->SetSourceConnection(cubeSource->GetOutputPort());
	glyph3Dmapper->SetInputData(polydata);
	glyph3Dmapper->Update();
	////

	//vtkNew<vtkNamedColors> colors;

	/*ofstream myfile;
	myfile.open("example.txt");
	myfile << points->GetNumberOfPoints();
	myfile.close();*/

	vtkSmartPointer<vtkActor> multiActor = vtkSmartPointer<vtkActor>::New();
	multiActor->SetMapper(glyph3Dmapper);
	multiActor->GetProperty()->EdgeVisibilityOn();
	multiActor->GetProperty()->SetOpacity(1);
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));

	//
	viewer->getRenderWindow()->GlobalWarningDisplayOff();
	//viewer->addCoordinateSystem(1);
	viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->AddActor(multiActor);
}



void calc_verdeckungen(CImg<int>& Verdeckungen, CImg<int>& Polygon_t, pcl::PointCloud<pcl::PointXYZI>& cloud,const float& step,const float& aufl) {

		pcl::PointXYZI minP, maxP;
		pcl::PointXYZ pt;
		pcl::RangeImage RI;
		create_RangeImage_from_pcl(RI, aufl, cloud);
		std::string range_file = "Color/RangeImage.png";
		//save_rangeImage_png(RI, range_file);


		// Verdeckungen

		pcl::getMinMax3D(cloud, minP, maxP);

		
		int xd = ceil((maxP.x - minP.x) / step);
		int yd = ceil((maxP.y - minP.y) / step);
		int zd = ceil((maxP.z - minP.z) / step);

		CImg<int> visu3(xd, yd, zd, 1, 0);

		for (int i = 1; i < cloud.size(); i++) {
			int xx = (cloud.points[i].x - minP.x) / step;
			int yy = (cloud.points[i].y - minP.y) / step;
			int zz = (cloud.points[i].z - minP.z) / step;
			if (xx >= 0 && yy >= 0 && zz >= 0) {
				visu3(xx, yy, zz) = 1;//%visu3(xx, yy) + 1;
			}
		}

		CImg<float> Temp = visu3.get_slice(0);
		////
		for (int i = 1; i <= zd - 1; i++) {
			Temp.operator+=(visu3.get_slice(i));
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr cl1(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i <= Temp.width() - 1; i++) {

			for (int j = 0; j <= Temp.height() - 1; j++) {
				if (Temp(i, j, 0) > 3) {
					pt.x = i;
					pt.y = j;
					pt.z = 0;
					cl1->push_back(pt);
				}
			}
		}


		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ConcaveHull<pcl::PointXYZ> chull;
		chull.setInputCloud(cl1);
		chull.setAlpha(100);
		chull.reconstruct(*cloud_hull);

		CImg<int> poly(cloud_hull->size(), 2);
		//std::vector<int> xm1;
		//std::vector<int> ym1;
		for (int i = 0; i < cloud_hull->size(); i++) {
			poly(i, 0) = cloud_hull->points[i].x;
			poly(i, 1) = cloud_hull->points[i].y;
		}
		int red[] = { 1 };

	//	CImg<int> Polygon_t;


		Polygon_t = CImg<int>(Temp.width(), Temp.height(), 1, 1, 0);

		Polygon_t.draw_polygon(poly, red);
		//Polygon_t.map(CImg<>::jet_LUT256());
		//Polygon_t.save_bmp("Color/Polygon.bmp");
		//	Polygon_t.display();
		
		calculate_Occlusions(RI, minP, maxP, step, Verdeckungen);


}

void crop_pcl(pcl::PointCloud<pcl::PointXYZI>& cloud_result, pcl::PointCloud<pcl::PointXYZI>& cloud,const pcl::PointXYZI minP,const pcl::PointXYZI maxP,const float step ) {

	pcl::CropBox<pcl::PointXYZI> boxFilter;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_t(new pcl::PointCloud<pcl::PointXYZI>);
	*cloud_t = cloud;
	boxFilter.setMin(Eigen::Vector4f(-100000, -100000, minP.z + zmin * step, 1.0));
	boxFilter.setMax(Eigen::Vector4f(100000, 100000, maxP.z + zmax * step, 1.0));
	boxFilter.setInputCloud(cloud_t);
	boxFilter.filter(cloud_result);
}