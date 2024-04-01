/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"
#include <pcl/filters/random_sample.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
    // Time Ransac process
    auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));

    bool my_solution = true;
    
    if (my_solution)
    {
        // My solution //////////////////////////////////////////////////////////////////////////////////////////
        // TODO: Fill in this function
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        std::map<int, std::unordered_set<int>> iterations_list;

        // For max iterations 
        for (int i = 0; i < maxIterations; i++) {

            pcl::RandomSample<pcl::PointXYZ> random_sample;
            pcl::Indices indices;


            // Randomly sample subset and fit line
            random_sample.setInputCloud(cloud);
            // Set the number of points to sample
            random_sample.setSample(4);

            // Optionally set the seed for the random function
            random_sample.setSeed(dist(gen)); // Seed value

            random_sample.filter(indices);
            pcl::PointXYZ point1 = cloud->points[indices[1]];
            pcl::PointXYZ point2 = cloud->points[indices[2]];
            pcl::PointXYZ point3 = cloud->points[indices[3]];

            indices.clear();

            // Calculate the coefficients of the line equation
            //float a = point1.y - point2.y;
            //float b = point2.x - point1.x;
            //float c = (point1.x * point2.y) - (point2.x * point1.y)

            // Calculate the coefficients of the plane equation
            float a = (point2.y - point1.y) * (point3.z - point1.z) - (point2.z - point1.z) * (point3.y - point1.y);
            float b = (point2.z - point1.z) * (point3.x - point1.x) - (point2.x - point1.x) * (point3.z - point1.z);
            float c = (point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x);
            float d = -(a * point1.x + b * point1.y + c * point1.z);

            // Create a pcl::ModelCoefficients object to store the line equation coefficients
            pcl::ModelCoefficients::Ptr line_coefficients(new pcl::ModelCoefficients());
            line_coefficients->values.push_back(a);
            line_coefficients->values.push_back(b);
            line_coefficients->values.push_back(c);
            line_coefficients->values.push_back(d);

            // Measure distance between every point and fitted line or plane
            for (int index = 0; index < cloud->size(); index++) {
                //float distance = std::abs(a * cloud->points[index].x + b * cloud->points[index].y + c) / std::sqrt(a * a + b * b);
                float distance = std::abs(a * cloud->points[index].x + b * cloud->points[index].y + c * cloud->points[index].z + d) / std::sqrt(a * a + b * b + c * c);
                // If distance is smaller than threshold count it as inlier
                if (distance < distanceTol) {
                    inliersResult.insert(index);
                }
            }
            iterations_list.emplace(i, inliersResult);
            inliersResult.clear();
        }

        int max_inliers = 0;
        int it_max_inliers = 0;
        for (int i = 0; i < iterations_list.size(); i++) {
            int max_inliers_tmp = iterations_list[i].size();
            if (max_inliers_tmp > max_inliers) {
                max_inliers = max_inliers_tmp;
                it_max_inliers = i;
            }
        }

        inliersResult = iterations_list[it_max_inliers];
    }
    // My Solution End ////////////////////////////////////////////////////////////////////////////////////////

    else {
        // Udacity Solution ///////////////////////////////////////////////////////////////////////////////////////
        while (maxIterations--)
        {
            // Randomly pick three points
            std::unordered_set<int> inliers;
            while (inliers.size() < 4)
                inliers.insert(rand() % (cloud->points.size()));

            float x1, y1, z1, x2, y2, z2, x3, y3, z3;
            
            auto itr = inliers.begin();
            x1 = cloud->points[*itr].x;
            y1 = cloud->points[*itr].y;
            z1 = cloud->points[*itr].z;
            itr++;
            x2 = cloud->points[*itr].x;
            y2 = cloud->points[*itr].y;
            z2 = cloud->points[*itr].z;
            itr++;
            x3 = cloud->points[*itr].x;
            y3 = cloud->points[*itr].y;
            z3 = cloud->points[*itr].z;

            float a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
            float b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
            float c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
            float d = -(a * x1 + b * y1 + c * z1);

            for (int index = 0; index < cloud->points.size(); index++)
            {
                if (inliers.count(index > 0))
                    continue;

                pcl::PointXYZ point = cloud->points[index];
                float x4 = point.x;
                float y4 = point.y;
                float z4 = point.z;

                float distance = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);

                if (distance <= distanceTol)
                    inliers.insert(index);
            }

            if (inliers.size() > inliersResult.size())
                inliersResult = inliers;
        }
    }
    // Udacity Solution End ///////////////////////////////////////////////////////////////////////////////////
   
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Ransac took " << elapsedTime.count() << " milliseconds" << std::endl;


	// Return indicies of inliers from fitted line with most inliers
	
	return inliersResult;

}

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();
	

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = Ransac(cloud, 100, 0.4);

	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
