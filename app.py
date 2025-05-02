from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
import rasterio
import os
import shutil
import tempfile
import base64
from typing import List
import zipfile
import logging

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
        
        geojson = gdf.to_json()
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
    """معالجة ملفات Raster (GeoTIFF) وتحويلها إلى base64 PNG."""
    logger.info(f"Step: Processing Raster: {raster_file}")
    try:
        with rasterio.open(raster_file) as src:
            data = src.read(1)
            from PIL import Image
            import numpy as np
            data = np.clip(data, np.percentile(data, 2), np.percentile(data, 98))
            data = (data - data.min()) / (data.max() - data.min()) * 255
            img = Image.fromarray(data.astype(np.uint8))
            from io import BytesIO
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
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
        # الخطوة 1: جمع أسماء الملفات المرفوعة
        uploaded_filenames = [file.filename.lower() for file in files]
        logger.info(f"Step: Uploaded filenames: {uploaded_filenames}")
        
        # الخطوة 2: التحقق من مكونات كل ملف .shp
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
        
        # الخطوة 3: حفظ جميع الملفات أولاً
        saved_files = []
        for file in files:
            logger.info(f"Step: Saving file: {file.filename}")
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            check_file_details(file_path)
        
        logger.info(f"Step: All files saved in temp_dir: {saved_files}")
        
        # الخطوة 4: معالجة الملفات بعد الحفظ
        for file in files:
            logger.info(f"Step: Processing file: {file.filename}")
            file_path = os.path.join(temp_dir, file.filename)
            
            if file.filename.lower().endswith('.zip'):
                # استخراج ملف ZIP
                extracted_files = extract_zip(file, temp_dir)
                shp_extracted_files = [f for f in extracted_files if f.lower().endswith('.shp')]
                for shp_file in shp_extracted_files:
                    results.append(process_shapefile(shp_file, temp_dir))
            elif file.filename.lower().endswith('.shp'):
                # معالجة ملفات Shapefile
                results.append(process_shapefile(file_path, temp_dir))
            elif file.filename.lower().endswith(('.tif', '.tiff')):
                # معالجة ملفات Raster
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

if __name__ == "__main__":
    logger.info("Step: Starting Uvicorn server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)