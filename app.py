from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
import os
import shutil
import tempfile
import base64
from typing import List
import zipfile
import logging
import numpy as np
from PIL import Image
import io
import json
from tools import clip_vector, save_temp_file, intersect_vectors, buffer_vector, near_features

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# إعداد CORS
logger.info("Step: Applying CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلد مؤقت للملفات المرفوعة
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_zip(zip_file: UploadFile, temp_dir: str) -> List[str]:
    """استخراج ملف ZIP إلى مجلد مؤقت وإرجاع قائمة الملفات."""
    logger.info(f"Step: Extracting ZIP file: {zip_file.filename}")
    zip_path = os.path.join(temp_dir, zip_file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)
    
    logger.info(f"Step: Verifying ZIP file existence: {zip_path}")
    if not os.path.exists(zip_path):
        logger.error(f"Step: ZIP file not saved: {zip_path}")
        raise ValueError(f"Failed to save ZIP file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    extracted_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
    logger.info(f"Step: Extracted files: {extracted_files}")
    return extracted_files

def check_file_details(file_path: str) -> tuple:
    """التحقق من وجود الملف وحجمه وإمكانية فتحه."""
    logger.info(f"Step: Checking file details for: {file_path}")
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        logger.info(f"Step: File {file_path} exists, size: {size} bytes")
        try:
            with open(file_path, 'rb') as f:
                f.read(1)  # محاولة قراءة بايت واحد للتحقق من الوصول
            logger.info(f"Step: File {file_path} can be opened")
            return True, size
        except Exception as e:
            logger.error(f"Step: Failed to open file {file_path}: {str(e)}")
            return False, 0
    else:
        logger.error(f"Step: File {file_path} does not exist")
        return False, 0

def check_shapefile_components(shp_file: str, temp_dir: str) -> dict:
    """التحقق من وجود ملفات .shx و.dbf المطلوبة لملف .shp، مع جعل .prj اختياريًا."""
    logger.info(f"Step: Checking Shapefile components for: {shp_file}")
    shp_base = os.path.splitext(os.path.basename(shp_file))[0]
    shx_file = os.path.join(temp_dir, f"{shp_base}.shx")
    shx_file_upper = os.path.join(temp_dir, f"{shp_base}.SHX")
    dbf_file = os.path.join(temp_dir, f"{shp_base}.dbf")
    prj_file = os.path.join(temp_dir, f"{shp_base}.prj")
    
    shx_exists, shx_size = check_file_details(shx_file)
    shx_upper_exists, _ = check_file_details(shx_file_upper)
    dbf_exists, dbf_size = check_file_details(dbf_file)
    prj_exists, prj_size = check_file_details(prj_file)
    
    if not (shx_exists or shx_upper_exists):
        logger.error("Step: No valid .shx file found")
        return {
            "success": False,
            "name": os.path.basename(shp_file),
            "error": f"Missing or inaccessible .shx file for {os.path.basename(shp_file)}. Ensure .shx is uploaded."
        }
    if not dbf_exists:
        logger.error("Step: No valid .dbf file found")
        return {
            "success": False,
            "name": os.path.basename(shp_file),
            "error": f"Missing or inaccessible .dbf file for {os.path.basename(shp_file)}. Ensure .dbf is uploaded."
        }
    
    if not prj_exists:
        logger.warning(f"Step: No .prj file found for {os.path.basename(shp_file)}. Will assume EPSG:4326 CRS.")
    
    logger.info(f"Step: Shapefile components verified: .shx size={shx_size} bytes, .dbf size={dbf_size} bytes, .prj={'present' if prj_exists else 'missing'}")
    return {
        "success": True,
        "shx_file": shx_file if shx_exists else shx_file_upper
    }

def process_shapefile(shp_file: str, temp_dir: str) -> dict:
    """معالجة ملفات Shapefile وتحويلها إلى GeoJSON."""
    logger.info(f"Step: Starting to process Shapefile: {shp_file}")
    
    # التحقق من وجود الملفات المطلوبة
    check_result = check_shapefile_components(shp_file, temp_dir)
    if not check_result["success"]:
        return check_result
    
    try:
        # ضبط خيار SHAPE_RESTORE_SHX
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        logger.info("Step: SHAPE_RESTORE_SHX set to YES")
        
        # التحقق من وجود ملف .shp
        shp_exists, shp_size = check_file_details(shp_file)
        if not shp_exists:
            logger.error("Step: .shp file not found or inaccessible")
            return {
                "success": False,
                "name": os.path.basename(shp_file),
                "error": f"Shapefile {os.path.basename(shp_file)} not found or inaccessible"
            }
        
        # قراءة Shapefile
        logger.info(f"Step: Reading Shapefile: {shp_file}")
        gdf = gpd.read_file(shp_file)
        logger.info(f"Step: Shapefile CRS: {gdf.crs}")
        
        # إذا لم يكن هناك CRS، عيّن EPSG:4326
        if gdf.crs is None:
            logger.info(f"Step: No CRS defined for {os.path.basename(shp_file)}. Setting CRS to EPSG:4326.")
            gdf = gdf.set_crs(epsg=4326)
        
        # إعادة الإسقاط إلى WGS84 إذا لزم الأمر
        if gdf.crs != "EPSG:4326":
            logger.info("Step: Reprojecting to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)
        
        # تحويل إلى GeoJSON وفحص الصحة
        geojson = gdf.to_json()
        try:
            json.loads(geojson)  # فحص صحة الـ JSON
            logger.info(f"Step: GeoJSON validated successfully for {os.path.basename(shp_file)}")
        except json.JSONDecodeError as e:
            logger.error(f"Step: Invalid GeoJSON for {os.path.basename(shp_file)}: {str(e)}")
            return {
                "success": False,
                "name": os.path.basename(shp_file),
                "error": f"Invalid GeoJSON format: {str(e)}"
            }
        
        logger.info(f"Step: Shapefile processed successfully: {os.path.basename(shp_file)}")
        return {
            "success": True,
            "name": os.path.basename(shp_file),
            "type": "vector",
            "geojson": geojson
        }
    except Exception as e:
        logger.error(f"Step: Failed to process Shapefile {shp_file}: {str(e)}")
        return {
            "success": False,
            "name": os.path.basename(shp_file),
            "error": str(e)
        }
    finally:
        # إزالة خيار SHAPE_RESTORE_SHX
        os.environ.pop('SHAPE_RESTORE_SHX', None)
        logger.info("Step: SHAPE_RESTORE_SHX unset")

def process_raster(raster_file: str) -> dict:
    """معالجة ملفات Raster (GeoTIFF) وتحويلها إلى base64 PNG مع تحويل الحدود إلى EPSG:4326."""
    logger.info(f"Step: Processing Raster: {raster_file}")
    try:
        with rasterio.open(raster_file) as src:
            logger.info(f"Number of bands in {raster_file}: {src.count}")
            if src.count >= 3:
                bands = src.read([1, 2, 3])
                bands = bands.transpose(1, 2, 0)
                arr = bands.astype(np.uint8)
                image = Image.fromarray(arr, mode='RGB')
            else:
                band = src.read(1)
                if len(band.shape) != 2:
                    band = band.squeeze()
                arr = band.astype(np.float32)
                if arr.max() == arr.min():
                    arr = np.zeros_like(arr)
                else:
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                arr = arr.astype(np.uint8)
                image = Image.fromarray(arr)

            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            logger.info(f"Step: Original bounds for {raster_file}: {bounds}")

            src_crs = src.crs
            logger.info(f"Step: Source CRS for {raster_file}: {src_crs}")
            if src_crs is None:
                logger.warning(f"Step: No CRS defined for {raster_file}. Assuming EPSG:32610 (UTM Zone 10N).")
                src_crs = "EPSG:32610"
            
            dst_crs = "EPSG:4326"
            transformed_bounds = transform_bounds(src_crs, dst_crs, bounds[0], bounds[1], bounds[2], bounds[3])
            bounds = [[transformed_bounds[1], transformed_bounds[0]], [transformed_bounds[3], transformed_bounds[2]]]
            logger.info(f"Step: Transformed bounds to EPSG:4326 for {raster_file}: {bounds}")

            logger.info(f"Step: Raster processed successfully: {os.path.basename(raster_file)}")
            return {
                "success": True,
                "name": os.path.basename(raster_file),
                "type": "raster",
                "data": img_str,
                "bounds": bounds
            }
    except Exception as e:
        logger.error(f"Step: Error processing Raster {raster_file}: {str(e)}")
        return {
            "success": False,
            "name": os.path.basename(raster_file),
            "error": str(e)
        }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    logger.info(f"Step: Received upload request with {len(files)} files")
    results = []
    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
        uploaded_filenames = [file.filename.lower() for file in files]
        logger.info(f"Step: Uploaded filenames: {uploaded_filenames}")
        
        shp_files = [f for f in uploaded_filenames if f.endswith('.shp')]
        for shp_filename in shp_files:
            shp_base = os.path.splitext(shp_filename)[0]
            required_extensions = [f"{shp_base}.shx", f"{shp_base}.dbf"]
            missing_extensions = [ext for ext in required_extensions if ext not in uploaded_filenames]
            if missing_extensions:
                logger.error(f"Step: Missing required Shapefile components for {shp_filename}: {missing_extensions}")
                results.append({
                    "success": False,
                    "name": shp_filename,
                    "error": f"Missing required Shapefile components: {missing_extensions}"
                })
                return JSONResponse(content=results)
        
        saved_files = []
        for file in files:
            logger.info(f"Step: Saving file: {file.filename}")
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            check_file_details(file_path)
        
        logger.info(f"Step: All files saved in temp_dir: {saved_files}")
        
        for file in files:
            logger.info(f"Step: Processing file: {file.filename}")
            file_path = os.path.join(temp_dir, file.filename)
            
            if file.filename.lower().endswith('.zip'):
                extracted_files = extract_zip(file, temp_dir)
                shp_extracted_files = [f for f in extracted_files if f.lower().endswith('.shp')]
                for shp_file in shp_extracted_files:
                    results.append(process_shapefile(shp_file, temp_dir))
            elif file.filename.lower().endswith('.shp'):
                results.append(process_shapefile(file_path, temp_dir))
            elif file.filename.lower().endswith(('.tif', '.tiff')):
                results.append(process_raster(file_path))
            else:
                logger.info(f"Step: Skipping file (not processed): {file.filename}")
                continue
    
    logger.info(f"Step: Returning response: {results}")
    response = JSONResponse(content=results)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    logger.info(f"Step: Response headers: {response.headers}")
    return response

@app.post("/clip")
async def clip_vector_endpoint(input_file: UploadFile = File(...), clip_file: UploadFile = File(...)):
    """نقطة نهاية لتقطيع طبقة متجهة باستخدام طبقة أخرى."""
    logger.info("Step: Received clip request")
    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
        input_path = save_temp_file(input_file, temp_dir)
        clip_path = save_temp_file(clip_file, temp_dir)
        result = clip_vector(input_path, clip_path)
        return JSONResponse(content=result)

@app.post("/intersect")
async def intersect_endpoint(files: List[UploadFile] = File(...)):
    """نقطة نهاية لتقاطع متعدد الطبقات."""
    logger.info(f"Step: Received intersect request with {len(files)} files")
    if len(files) < 2:
        return JSONResponse(content={"success": False, "error": "At least two layers are required."})
    
    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
        file_paths = [save_temp_file(file, temp_dir) for file in files]
        result = intersect_vectors(file_paths)
        return JSONResponse(content=result)

@app.post("/buffer")
async def buffer_endpoint(input_file: UploadFile = File(...), buffer_distance: float = 100, unit: str = "Meters"):
    """نقطة نهاية لإنشاء منطقة تأثير."""
    logger.info("Step: Received buffer request")
    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
        input_path = save_temp_file(input_file, temp_dir)
        result = buffer_vector(input_path, buffer_distance, unit)
        return JSONResponse(content=result)

@app.post("/near")
async def near_endpoint(input_file: UploadFile = File(...), near_file: UploadFile = File(...), max_distance: float = None, k_neighbors: int = 1):
    """نقطة نهاية لإيجاد أقرب المعالم."""
    logger.info("Step: Received near request")
    with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
        input_path = save_temp_file(input_file, temp_dir)
        near_path = save_temp_file(near_file, temp_dir)
        result = near_features(input_path, near_path, max_distance, k_neighbors)
        return JSONResponse(content=result)

if __name__ == "__main__":
    logger.info("Step: Starting Uvicorn server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)