import sys
import os
import torch

def load_cuda():
    
    if torch.cuda.is_available():
        print("✓ CUDA is available")
    else:
        print("Loading HPC cluster CUDA module...")
    
        try: 
            sys.path.append("/usr/share/lmod/lmod/init")
            from env_modules_python import module
        
            cuda_root = "/software/easybuild/el8/amd_zen3/all/CUDA/12.6.0"

            module("load", "CUDA")

            os.environ["PATH"] = f"{cuda_root}/bin:{os.environ['PATH']}"
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
            os.environ["CUDA_HOME"] = cuda_root
            os.environ["CUDA_PATH"] = cuda_root
            
            if torch.cuda.is_available():
                print("✓ CUDA is available now after loading module.")
            else:
                print("✗ CUDA is still not available after loading module.")

        except Exception as e:
            print("Error loading HPC cluster CUDA module:", e)
            print("Proceeding without loading CUDA module.")
        
    
    
def load_gdal():

    print("Loading HPC cluster GDAL module...")
    try: 
        sys.path.append("/usr/share/lmod/lmod/init")
        from env_modules_python import module
    except Exception as e:
        print("Error importing env_modules_python:", e)
        return
    
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