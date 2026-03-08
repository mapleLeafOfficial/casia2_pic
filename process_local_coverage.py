import os
import shutil

def organize_coverage_data():
    source_dir = r"D:\code-work\开题报告final\基于深度学习的图像篡改检测方法研究与应用\OneDrive_2_2026-3-8"
    casia_dir = "data/raw/CASIA2"
    au_dir = os.path.join(casia_dir, "Au")
    tp_dir = os.path.join(casia_dir, "Tp")
    mask_dir = os.path.join(casia_dir, "mask")
    
    print("Preparing directories...")
    for d in [au_dir, tp_dir, mask_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        
    cov_img_dir = os.path.join(source_dir, "image")
    cov_mask_dir = os.path.join(source_dir, "mask")
    
    count_au = 0
    count_tp = 0
    
    for img_name in os.listdir(cov_img_dir):
        if not img_name.endswith(('.tif', '.jpg', '.png')):
            continue
            
        src_img = os.path.join(cov_img_dir, img_name)
        basename, ext = os.path.splitext(img_name)
        
        if basename.endswith('t'):
            # It's a tampered image
            dst_img = os.path.join(tp_dir, f"CM_{basename}{ext}")
            shutil.copy2(src_img, dst_img)
            
            # Find its mask (name is Xforged.tif)
            base_id = basename[:-1] # Remove 't'
            mask_name = f"{base_id}forged.tif"
            src_mask = os.path.join(cov_mask_dir, mask_name)
            
            if os.path.exists(src_mask):
                dst_mask = os.path.join(mask_dir, f"CM_{basename}.tif")
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Warning: Mask {mask_name} not found for {img_name}")
                
            count_tp += 1
        else:
            # It's an authentic image
            dst_img = os.path.join(au_dir, f"Au_{basename}{ext}")
            shutil.copy2(src_img, dst_img)
            count_au += 1
            
    print(f"Successfully processed {count_au} Authentic images and {count_tp} Tampered images.")

if __name__ == "__main__":
    organize_coverage_data()
