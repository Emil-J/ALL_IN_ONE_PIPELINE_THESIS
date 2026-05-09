canvas = iface.mapCanvas()
extent = canvas.extent()
center = extent.center()

# Project CRS
project_crs = canvas.mapSettings().destinationCrs()

# Transform to WGS84
wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
transform = QgsCoordinateTransform(project_crs, wgs84, QgsProject.instance())

center_wgs84 = transform.transform(center)

print("EPSG:25832:", center.x(), center.y())
print("EPSG:4326:", center_wgs84.y(), center_wgs84.x())
