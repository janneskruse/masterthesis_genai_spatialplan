import sys
import os
sys.path.append("/usr/share/lmod/lmod/init")
from env_modules_python import module

def load_cuda():

    cuda_root = "/software/easybuild/el8/amd_zen3/all/CUDA/12.6.0"

    module("load", "CUDA")

    os.environ["PATH"] = f"{cuda_root}/bin:{os.environ['PATH']}"
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["CUDA_HOME"] = cuda_root
    os.environ["CUDA_PATH"] = cuda_root
    
    
def load_gdal():

    gdal_root = "/software/easybuild/el8/amd_zen3/all/GDAL/3.9.0-foss-2023b"
    module("load", "GDAL")
    
    # GDAL setup with corrected paths
    os.environ["PATH"] = f"{gdal_root}/bin:{os.environ['PATH']}"
    os.environ["LD_LIBRARY_PATH"] = f"{gdal_root}/lib64:{gdal_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["GDAL_DATA"] = f"{gdal_root}/share/gdal"
    os.environ["GDAL_DRIVER_PATH"] = f"{gdal_root}/lib64/gdalplugins"
    print("GDAL is available: ", os.system("gdalinfo --version")) # Test GDAL
    
    # os.environ["PATH"] = f"{gdal_root}/bin:" + os.environ["PATH"]
    # os.environ["LD_LIBRARY_PATH"] = f"{gdal_root}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    # os.environ["GDAL_DATA"] = f"{gdal_root}/share/gdal"
    # os.environ["GDAL_DRIVER_PATH"] = f"{gdal_root}/lib/gdalplugins"