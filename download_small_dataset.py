import os
import urllib.request
import zipfile
import shutil

def download_and_extract_coverage():
    url = "https://github.com/wenbihan/coverage/archive/refs/heads/master.zip"
    zip_path = "coverage.zip"
    extract_path = "coverage-master"
    
    print("Downloading COVERAGE dataset (Small image forgery dataset, ~50MB)...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    print("Organizing into CASIA2 format structure for training...")
    casia_dir = "data/raw/CASIA2"
    au_dir = os.path.join(casia_dir, "Au")
    tp_dir = os.path.join(casia_dir, "Tp")
    mask_dir = os.path.join(casia_dir, "mask")
    
    os.makedirs(au_dir, exist_ok=True)
    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # COVERAGE has image and mask folders
    # Images are in coverage-master/image (e.g. 1.tif, 1t.tif)
    # Masks are in coverage-master/mask (e.g. 1forgery.tif)
    cov_img_dir = os.path.join(extract_path, "image")
    cov_mask_dir = os.path.join(extract_path, "mask")
    
    if os.path.exists(cov_img_dir):
        for img_name in os.listdir(cov_img_dir):
            if not img_name.endswith(('.tif', '.jpg', '.png')):
                continue
                
            src_img = os.path.join(cov_img_dir, img_name)
            
            # Tampered image has 't' before the extension, e.g., '1t.tif'
            basename, ext = os.path.splitext(img_name)
            if basename.endswith('t'):
                # It's a tampered image.
                # Project datasets.py expects "CM" or "Sp" in name to recognize tamper type.
                # COVERAGE is Copy-Move, so we add CM.
                dst_img = os.path.join(tp_dir, f"CM_{basename}{ext}")
                shutil.copy2(src_img, dst_img)
                
                # Check for corresponding mask
                # The mask for '1t.tif' is usually '1forgery.tif'
                base_id = basename[:-1]
                mask_name = f"{base_id}forgery.tif"
                src_mask = os.path.join(cov_mask_dir, mask_name)
                if os.path.exists(src_mask):
                    # Destination mask name should match image name without extension to be found by datasets.py
                    # Based on datasets.py: f"{stem}.png" or similar. We save as png or tif.
                    dst_mask = os.path.join(mask_dir, f"CM_{basename}.tif")
                    shutil.copy2(src_mask, dst_mask)
            else:
                # It's an authentic image.
                dst_img = os.path.join(au_dir, f"Au_{basename}{ext}")
                shutil.copy2(src_img, dst_img)
                
    print("Dataset organized successfully!")
    
    # Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

if __name__ == "__main__":
    download_and_extract_coverage()
