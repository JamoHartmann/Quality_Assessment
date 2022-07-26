# Quality_Assessment
 terrestrial laserscanning (TLS) Quality assessment including:
 1. 3D local precision (variance covariance propagation of uncertanties of the tls- observations (two angle one distance) -> distance precision is calculated by the intensity based model) 
 2. 3D global precision (variance covariance propagation of uncertainties of transformation parameters)
 3. Incidence angle 
 4. Spot size
 5. Edge points
 
 Visual studio solution including all necessary depedencies:
 
 Example Folder includes:
 1. Programm.bat: Batch File to run the .exe -> Input cloud, as well as parameters are specified in this file
 2. 3.e57: Point cloud in e57-Format: The Software only works with e57-Format
 3. Scanner ++ : Scanner parameters 
 4. VCV_3: Variance covariance matrix for the Transformation (Transformation matrix is known from e57-Format). Order of the Values is important. It is the same order as the Scantra output.
 
