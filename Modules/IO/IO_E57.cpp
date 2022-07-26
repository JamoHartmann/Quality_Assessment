#include <IO_E57.h>


namespace IO_E57toPCD {
	void Pcl_type(bool& rgb, bool& intensity, bool& xyz, bool& org, e57::ustring file) {
		e57::Reader eReader(file);
		int 		scanIndex = 0;	//picking the first scan
		e57::Data3D	scanHeader;		//read scan's header information
		eReader.ReadData3D(scanIndex, scanHeader);
		rgb = scanHeader.pointFields.intensityField;
		intensity = scanHeader.pointFields.colorGreenField;
		xyz = scanHeader.pointFields.cartesianXField;
		int64_t nColumn = 0;		//Number of Columns in a structure scan (from "indexBounds" if structure data)
		int64_t nRow = 0;			//Number of Rows in a structure scan	
		int64_t nPointsSize = 0;	//Number of points 
		int64_t nGroupsSize = 0;	//Number of groups (from "groupingByLine" if present)
		int64_t nCountsSize = 0;	//Number of points per group
		int64_t  countSize = 0;
		bool columnIndex = 0;

		eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, countSize, columnIndex);

		org = columnIndex;
		eReader.Close();
	}




	void read_e57_XYZI(pcl::PointCloud<pcl::PointXYZI>& cloud, e57::ustring file, const bool& org) {

		e57::Reader eReader(file);
		int 		scanIndex = 0;	//picking the first scan
		e57::Data3D	scanHeader;		//read scan's header information
		eReader.ReadData3D(scanIndex, scanHeader);
		int64_t nColumn = 0;		//Number of Columns in a structure scan (from "indexBounds" if structure data)
		int64_t nRow = 0;			//Number of Rows in a structure scan	
		int64_t nPointsSize = 0;	//Number of points 
		int64_t nGroupsSize = 0;	//Number of groups (from "groupingByLine" if present)
		int64_t nCountsSize = 0;	//Number of points per group
		int64_t  countSize = 0;
		bool columnIndex = 0;
		eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, countSize, columnIndex);
		int64_t nSize = (nRow > 0) ? nRow : 1024;//Pick a size for buffers
		//if (nSize > 5000) { nSize = 250; }
		float* xData = new float[nSize];
		float* yData = new float[nSize];
		float* zData = new float[nSize];
		int8_t* cartesianInvalidState = new int8_t[nSize];
		float* intensity = new float[nSize];
		float* intens2 = new float[nSize];
		uint8_t* cr = new uint8_t[nSize];
		uint8_t* cg = new uint8_t[nSize];
		uint8_t* cb = new uint8_t[nSize];
		int64_t* id = new int64_t[nSize];

		int64_t* start = new int64_t[nSize];
		int64_t* pointCount = new int64_t[nSize];
		int32_t* rowIndex = new int32_t[nSize];
		int32_t* columnIndex2 = new int32_t[nSize];
		bool tt = eReader.ReadData3DGroupsData(scanIndex, nSize, id, start, pointCount);
		e57::Data3DPointsData bu = { xData,yData,zData,cartesianInvalidState,intensity,NULL,cr,cg,cb,NULL,NULL,NULL,NULL,NULL,rowIndex,columnIndex2,NULL,NULL,NULL,NULL,NULL,NULL,NULL,intens2 };

		e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData(scanIndex, nSize, bu);

		double redOffset = scanHeader.colorLimits.colorRedMinimum;
		double	redRange = scanHeader.colorLimits.colorRedMaximum - redOffset;
		double greenOffset = scanHeader.colorLimits.colorGreenMinimum;
		double	greenRange = scanHeader.colorLimits.colorGreenMaximum - redOffset;
		double blueOffset = scanHeader.colorLimits.colorBlueMinimum;
		double	blueRange = scanHeader.colorLimits.colorBlueMaximum - redOffset;

		std::cout << nGroupsSize << " " << countSize << std::endl;


		cloud.sensor_origin_ = Eigen::Vector4f(scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z, 1);
		Eigen::Quaternionf Quater_rot = Eigen::Quaternionf(scanHeader.pose.rotation.w, scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z);
		cloud.sensor_orientation_ = Quater_rot;
		unsigned long size = 0;
		int temp = 0;
		float p_radius, az, elev;
		int percentage;
		int tt1 = 0;
		int N_Points = 0;


		if (org == true) {
			cloud.width = nGroupsSize;
			cloud.height = countSize;
			cloud.is_dense = false;
			cloud.resize(cloud.width * cloud.height);
			while ((size = dataReader.read()) > 0)	//Each call to dataReader.read() will retrieve the next column of data.
			{

				for (unsigned long i = 0; i < size; i = i + 1)		//x,y,z Data buffers have the next column of data.
				{
					if (xData[i] != 0 && yData[i] != 0 && zData[i] != 0) {
						N_Points = N_Points + 1;
						int tc = ceil(columnIndex2[i]);
						int tr = ceil(rowIndex[i]);
						cloud.at(tc, tr).x = xData[i];
						cloud.at(tc, tr).y = yData[i];
						cloud.at(tc, tr).z = zData[i];
						cloud.at(tc, tr).intensity = intensity[i];

					}
					else {
						int tc = ceil(columnIndex2[i]);
						int tr = ceil(rowIndex[i]);
						cloud.at(tc, tr).x = NAN;
						cloud.at(tc, tr).y = NAN;
						cloud.at(tc, tr).z = NAN;
					}

					temp = temp + 1;
				}
			}
		}


		if (org == false) {
			cloud.width = nPointsSize;
			cloud.height = 1;
			cloud.is_dense = false;
			cloud.resize(cloud.width * cloud.height);
			while ((size = dataReader.read()) > 0)	//Each call to dataReader.read() will retrieve the next column of data.
			{

				for (unsigned long i = 0; i < size; i = i + 1)		//x,y,z Data buffers have the next column of data.
				{
					if (xData[i] != 0 && yData[i] != 0 && zData[i] != 0) {
						cloud.points[temp].x = xData[i];
						cloud.points[temp].y = yData[i];
						cloud.points[temp].z = zData[i];
						cloud.points[temp].intensity = intensity[i];
					}
					temp = temp + 1;
				}
			}
		}

	}
}