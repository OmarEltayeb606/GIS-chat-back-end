from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import rasterio
import os
import shutil
import tempfile
import base64
import numpy as np
from PIL import Image
import io
import logging
import pyproj

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# دالة لتحويل المصفوفة إلى Base64
def array_to_base64(arr, is_rgb=False):
    logger.info(f"Array shape before processing: {arr.shape}")
    
    if is_rgb:
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB array with shape (height, width, 3), got {arr.shape}")
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr, mode='RGB')
    else:
        if len(arr.shape) != 2:
            arr = arr.squeeze()
            logger.info(f"Array shape after squeeze: {arr.shape}")
            if len(arr.shape) != 2:
                raise ValueError(f"Cannot convert array with shape {arr.shape} to image.")
        arr = arr.astype(np.float32)
        if arr.max() == arr.min():
            logger.warning("Array has no variation in values, setting to zero.")
            arr = np.zeros_like(arr)
        else:
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr)

    # تغيير حجم الصورة لتقليل حجم البيانات (اختياري)
    image = image.resize((500, 500), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# دالة محسّنة لتخمين نظام الإسقاط
def guess_crs_from_bounds(bounds):
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    logger.info(f"Bounds for CRS guessing: left={left}, bottom={bottom}, right={right}, top={top}")

    # التحقق مما إذا كانت الإحداثيات تبدو وكأنها Lat/Lng
    if -180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90:
        logger.info("Bounds appear to be in Lat/Lng (EPSG:4326).")
        return "EPSG:4326"

    # إذا كانت الإحداثيات كبيرة (مثل قيم UTM بالأمتار)
    if 0 <= left <= 1_000_000 and 0 <= right <= 1_000_000:
        # افترض أن الإحداثيات في UTM، لكن ليس لدينا Zone محدد بعد
        # نحتاج إلى تخمين الـ Zone بناءً على تقدير خط الطول (Longitude)
        # نستخدم Zone 35N مؤقتًا لتحويل الإحداثيات إلى Lat/Lng
        try:
            # نستخدم Zone 35N كقاعدة أساسية لتحويل الإحداثيات
            utm_to_latlon = pyproj.Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
            avg_x = (left + right) / 2
            avg_y = (bottom + top) / 2
            lon, lat = utm_to_latlon.transform(avg_x, avg_y)

            # التحقق من صحة الإحداثيات الناتجة
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                logger.warning("Invalid Lat/Lng coordinates after transformation. Trying a different approach...")
                # إذا فشل التحويل، نستخدم طريقة تقدير مبسطة
                # تقدير خط الطول بناءً على قيم easting (افتراض false easting 500000 في UTM)
                approx_lon = ((avg_x - 500000) / 111320)  # 111320 متر لكل درجة تقريبًا
                zone = int((approx_lon + 180) / 6) + 1
            else:
                # استخدام خط الطول المحسوب لتحديد الـ Zone
                zone = int((lon + 180) / 6) + 1

            # تحديد نصف الكرة الأرضية
            if lat >= 0:
                crs_code = f"EPSG:326{zone:02d}"  # UTM شمالي
            else:
                crs_code = f"EPSG:327{zone:02d}"  # UTM جنوبي
            logger.info(f"Calculated UTM zone: {zone}, CRS: {crs_code}")
            return crs_code
        except Exception as e:
            logger.warning(f"Error guessing UTM zone: {str(e)}. Falling back to default.")
            return "EPSG:32635"
    else:
        logger.warning("Bounds do not match UTM or Lat/Lng. Assuming EPSG:32635 as fallback.")
        return "EPSG:32635"

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_name = file.filename
            file_path = os.path.join(temp_dir, file_name)

            logger.info(f"Saving temporary file: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            try:
                if file_name.lower().endswith(('.tif', '.tiff')):
                    logger.info(f"Processing TIFF: {file_name}")
                    with rasterio.open(file_path) as dataset:
                        logger.info(f"Number of bands in {file_name}: {dataset.count}")
                        logger.info(f"Width: {dataset.width}, Height: {dataset.height}")
                        logger.info(f"Metadata: {dataset.meta}")

                        # قراءة البيانات
                        if dataset.count >= 3:
                            if dataset.count >= 4 and "L1TP" in file_name:
                                bands = dataset.read([4, 3, 2])
                                logger.info(f"RGB bands shape (Landsat): {bands.shape}")
                            else:
                                bands = dataset.read([1, 2, 3])
                                logger.info(f"RGB bands shape (Generic): {bands.shape}")
                            bands = bands.transpose(1, 2, 0)
                            logger.info(f"RGB bands shape after transpose: {bands.shape}")
                            base64_data = array_to_base64(bands, is_rgb=True)
                        else:
                            band = dataset.read(1)
                            logger.info(f"Band shape: {band.shape}")
                            if len(band.shape) != 2:
                                band = band.squeeze()
                                logger.info(f"Band shape after squeeze: {band.shape}")
                                if len(band.shape) != 2:
                                    raise ValueError(f"Unexpected band shape {band.shape} after squeeze.")
                            base64_data = array_to_base64(band, is_rgb=False)

                        # استخراج الحدود والنظام الإسقاطي
                        bounds = dataset.bounds
                        crs = dataset.crs
                        logger.info(f"Original Bounds for {file_name}: {[[bounds.bottom, bounds.left], [bounds.top, bounds.right]]}")
                        logger.info(f"CRS for {file_name}: {crs}")

                        # تحديد نظام الإسقاط (CRS)
                        crs_str = str(crs).upper() if crs else None
                        if not crs_str or not crs_str.startswith('EPSG:'):
                            logger.warning(f"No valid CRS found for {file_name}. Attempting to guess CRS...")
                            crs_str = guess_crs_from_bounds(bounds)
                            if not crs_str:
                                logger.warning(f"Could not guess CRS for {file_name}. Assuming EPSG:32635 (UTM Zone 35N).")
                                crs_str = "EPSG:32635"
                            else:
                                logger.info(f"Guessed CRS for {file_name}: {crs_str}")

                        results.append({
                            "success": True,
                            "name": file_name,
                            "type": "raster",
                            "data": base64_data,
                            "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            "crs": crs_str
                        })

                else:
                    results.append({
                        "success": False,
                        "name": file_name,
                        "error": "نوع ملف غير مدعوم."
                    })

            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
                results.append({
                    "success": False,
                    "name": file_name,
                    "error": f"خطأ في معالجة {file_name}: {str(e)}"
                })

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)