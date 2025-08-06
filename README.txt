Description
This dataset includes supplemental weather and mapping data collected for the FLAME project. Links and access to other potentially helpful external data are provided as well. The objective of this dataset is to provide users of the FLAME primary data context, location, and other supplementary data that may be used for analysis. 

Data includes: Bluestem prescribed fire burn plan, Light Detection and Ranging (LiDAR) and Digital Elevation Model (DEM) point clouds, georeferenced orthomosaic and color point cloud, and spot weather forecast


Size: 
4.7 GB


Platform:
Data is accessible on any platform capable of reading the file formats above.


Environment:
Linux, Mac, and Windows operating systems are all capable of reading and viewing the files.


Major Component Description:
Burn plan and Weather-
An incident action plan (IAP) was issued to all fire resources that were part of the Bluestem prescribed fire. Includes maps, objectives, and resources.

IAP: blustm_burnplan.pdf

A spot weather forecast was included in the Bluestem Rx Incident Action Plan. Issued by the National Weather Service, the forecast is specific to the burn area for 24 hours.

Spot weather forecast: blustm_wx_spot.jpg

Weather can also be accessed from Remote Automated Weather Stations (RAWs). These stations collect hourly weather all over the Western US for monitoring fire weather conditions.

Learn more about RAWS: https://raws.nifc.gov

Access RAWS data: https://raws.dri.edu

LiDAR and DEM-
The Coconino and Kaibab National Forests have aerial LiDAR-derived point clouds and Digital Elevation Models for the entire forest, flown in 2018 and 2019. Another flight is expected in Fall 2022. 

Learn more about LiDAR and where you can download: https://www.usgs.gov/faqs/what-lidar-data-and-where-can-i-download-it

Access point cloud and DEM datasets from USGS LidarExplorer or National Map Download Client: https://prd-tnm.s3.amazonaws.com/LidarExplorer/index.html#/
https://apps.nationalmap.gov/downloader/#/

Georeferenced orthomosaic and color point cloud-
DJI_001 and DJI_003 pre-burn videos were used to make a color point cloud and orthomosaic via structure from motion. Videos were parsed into images every 60 frames and aligned to build a point cloud in Agisoft Metashape. The Point cloud and orthomosaic are georeferenced with natural ground control points and satellite imagery cross-referencing (google maps) with mean error of 0.705 meters. Orthomosaic resolution is ~ 7.6 cm/pixel. Coordinate system is WGS84 (EPSG:4326). 

Software used: Agisoft Metashape Version 1.8.1

Notable Parameters:
Align
Quality: “high”
Generic preselection: “Yes”

Point cloud
Quality: “high”
Depth Filtering: “Low”

DEM (intermediate to orthomosaic)
Source: point cloud
Quality: “High”

Orthomosaic
Blending: “mosaic”
 
Point cloud: blustm_pointcloud_RGB.laz
 
Orthomosaic: blustm_ortho_RGB.jpg

Video: DJI_0001.MP4 and DJI_0003.MP4


Contact information:
Abolfazl Razi, Clemson University, arazi@clemson.edu
Fatemeh Afghah, Clemson University, fafghah@clemson.edu
Leo O'Neill, Northern Arizona University, leo.oneill@nau.edu
