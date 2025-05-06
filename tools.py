import geopandas as gpd
import json
import tempfile
import os
import shutil
import logging
from functools import reduce
import folium
from shapely.geometry import shape, mapping
from tqdm import tqdm
from folium.plugins import MarkerCluster

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_temp_file(upload_file, temp_dir: str) -> str:
    """حفظ ملف مرفوع في دليل مؤقت."""
    file_path = os.path.join(temp_dir, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

def clip_vector(input_file_path: str, clip_file_path: str) -> dict:
    """تقطيع طبقة متجهة باستخدام طبقة تقطيع."""
    try:
        input_gdf = gpd.read_file(input_file_path)
        clip_gdf = gpd.read_file(clip_file_path)
        
        if input_gdf.crs != clip_gdf.crs:
            logger.info("Reprojecting clip layer to match input CRS")
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)
        
        clipped = gpd.clip(input_gdf, clip_gdf)
        geojson = json.loads(clipped.to_json())
        
        logger.info("Clip operation completed successfully")
        return {"success": True, "geojson": geojson}
    except Exception as e:
        logger.error(f"Error in clip operation: {str(e)}")
        return {"success": False, "error": str(e)}

def intersect_vectors(input_files: list, output_path: str = None) -> dict:
    """تقاطع متعدد الطبقات المتجهة."""
    try:
        if len(input_files) < 2:
            return {"success": False, "error": "Please select at least two layers."}

        # تحميل الطبقات وحل مشكلة CRS
        layers = [gpd.read_file(f) for f in input_files]
        base_crs = layers[0].crs
        for i, gdf in enumerate(layers):
            if gdf.crs != base_crs:
                layers[i] = gdf.to_crs(base_crs)

        # إعادة تسمية الأعمدة لتجنب التعارض
        for i, gdf in enumerate(layers):
            gdf.columns = [f"{col}_{i}" if col != "geometry" else col for col in gdf.columns]

        # التحقق من الأشكال الهندسية
        for gdf in layers:
            if gdf.is_empty.any():
                raise ValueError("One or more layers contain empty geometries.")

        # تنفيذ التقاطع
        result = reduce(lambda l, r: gpd.overlay(l, r, how='intersection'), layers)

        # تحويل إلى GeoJSON
        geojson = json.loads(result.to_json())

        if output_path:
            result.to_file(output_path)
            logger.info(f"Intersect result saved to: {output_path}")

        return {"success": True, "geojson": geojson}
    except Exception as e:
        logger.error(f"Error in intersect operation: {str(e)}")
        return {"success": False, "error": str(e)}

def buffer_vector(input_file_path: str, buffer_distance: float, unit: str = "Meters", output_path: str = None) -> dict:
    """إنشاء منطقة تأثير (buffer) حول طبقة متجهة."""
    try:
        unit_multipliers = {"Meters": 1, "Kilometers": 1000, "Miles": 1609.34}
        distance_in_meters = buffer_distance * unit_multipliers.get(unit, 1)

        gdf = gpd.read_file(input_file_path)
        buffered = gdf.copy()
        buffered['geometry'] = buffered.geometry.buffer(distance_in_meters)

        geojson = json.loads(buffered.to_json())

        if output_path:
            buffered.to_file(output_path)
            logger.info(f"Buffer result saved to: {output_path}")

        return {"success": True, "geojson": geojson}
    except Exception as e:
        logger.error(f"Error in buffer operation: {str(e)}")
        return {"success": False, "error": str(e)}

def near_features(input_file_path: str, near_file_path: str, max_distance: float = None, k_neighbors: int = 1, output_path: str = None) -> dict:
    """إيجاد أقرب المعالم باستخدام GeoPandas."""
    try:
        input_gdf = gpd.read_file(input_file_path)
        near_gdf = gpd.read_file(near_file_path)

        if input_gdf.crs != near_gdf.crs:
            raise ValueError("CRS of input and near layers do not match!")

        if input_gdf.crs.is_geographic:
            logger.info("Reprojecting to EPSG:3857 for accurate distance calculations")
            input_gdf = input_gdf.to_crs(epsg=3857)
            near_gdf = near_gdf.to_crs(epsg=3857)

        if 'FID' not in near_gdf.columns:
            near_gdf['FID'] = range(1, len(near_gdf) + 1)

        near_sindex = near_gdf.sindex

        def find_nearest(row):
            possible_matches_index = list(near_sindex.nearest(row.geometry.bounds, num_results=k_neighbors))
            possible_matches = near_gdf.iloc[possible_matches_index]
            possible_matches = possible_matches.copy()
            possible_matches['distance'] = possible_matches.geometry.distance(row.geometry)
            possible_matches = possible_matches.sort_values('distance')

            if max_distance is not None:
                possible_matches = possible_matches[possible_matches['distance'] <= max_distance]

            if not possible_matches.empty:
                nearest = possible_matches.iloc[0]
                return pd.Series({'NEAR_FID': nearest['FID'], 'NEAR_DIST': nearest['distance']})
            return pd.Series({'NEAR_FID': None, 'NEAR_DIST': None})

        input_gdf[['NEAR_FID', 'NEAR_DIST']] = input_gdf.apply(find_nearest, axis=1)

        geojson = json.loads(input_gdf.to_json())

        if output_path:
            input_gdf.to_file(output_path)
            logger.info(f"Near analysis result saved to: {output_path}")

        return {"success": True, "geojson": geojson}
    except Exception as e:
        logger.error(f"Error in near operation: {str(e)}")
        return {"success": False, "error": str(e)}