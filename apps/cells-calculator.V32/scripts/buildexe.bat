rem pyinstaller --noconsole --hidden-import=rasterio.sample --hidden-import=rasterio.vrt --hidden-import=rasterio.features --hidden-import=rasterio.warp  --hidden-import=xsdata_pydantic_basemodel.hooks --add-data "modelconfig.json;." --add-data "model/*.pt*;model" --add-data "model/*.onnx;model" main.py
rem pyinstaller exe-main.spec
rem robocopy .\trainedmodels .\dist\main\trainedmodels

