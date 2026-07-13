import os
import shutil
import random
import yaml
import zipfile

def main():
    # 1. การตั้งค่าโฟลเดอร์ต้นทาง
    source_img_dir = "images"
    source_lbl_dir = "labels"
    
    # 2. การตั้งค่าโฟลเดอร์ปลายทางสำหรับ YOLO
    dataset_dir = "yolo_dataset"
    train_img_path = os.path.join(dataset_dir, "train", "images")
    train_lbl_path = os.path.join(dataset_dir, "train", "labels")
    val_img_path = os.path.join(dataset_dir, "val", "images")
    val_lbl_path = os.path.join(dataset_dir, "val", "labels")
    
    # ล้างโฟลเดอร์เดิมถ้ามีอยู่ เพื่อป้องกันไฟล์ซ้ำซ้อน
    if os.path.exists(dataset_dir):
        print(f"กำลังล้างโฟลเดอร์เดิม '{dataset_dir}'...")
        shutil.rmtree(dataset_dir)
        
    # สร้างโครงสร้างโฟลเดอร์ใหม่ทั้งหมด
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_lbl_path, exist_ok=True)

    # 3. ดึงรายชื่อรูปภาพทั้งหมดที่มีในโฟลเดอร์ images
    if not os.path.exists(source_img_dir):
        print(f"Error: ไม่พบโฟลเดอร์ '{source_img_dir}' กรุณาเก็บข้อมูลก่อนรันสคริปต์นี้")
        return
        
    all_images = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(all_images) == 0:
        print("Error: ไม่พบไฟล์รูปภาพในโฟลเดอร์ images/")
        return
        
    print(f"พบไฟล์รูปภาพทั้งหมด {len(all_images)} ไฟล์")

    # 4. ตรวจหาคลาสที่มีทั้งหมดจากไฟล์ .txt ใน labels
    detected_classes = set()
    valid_pairs = []
    
    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        lbl_file_path = os.path.join(source_lbl_dir, lbl_name)
        
        if os.path.exists(lbl_file_path):
            valid_pairs.append((img_name, lbl_name))
            # อ่านคลาสจากในไฟล์พิกัด
            try:
                with open(lbl_file_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            detected_classes.add(int(parts[0]))
            except Exception as e:
                print(f"Warning: อ่านไฟล์ {lbl_name} ไม่สำเร็จ: {e}")
        else:
            print(f"Warning: รูปภาพ '{img_name}' ไม่มีไฟล์พิกัด '{lbl_name}' (ข้ามภาพนี้)")

    if len(valid_pairs) == 0:
        print("Error: ไม่พบภาพที่จับคู่กับไฟล์ .txt ได้เลย")
        return

    print(f"จับคู่ภาพและพิกัดสำเร็จทั้งหมด {len(valid_pairs)} คู่")
    print(f"คลาสที่ตรวจพบทั้งหมด: {sorted(list(detected_classes))}")

    # 5. สุ่มแบ่งข้อมูล (Train 80% / Val 20%)
    random.seed(42)  # ตั้งค่า Seed เพื่อให้สุ่มได้แบบเดิมทุกครั้งที่รัน
    random.shuffle(valid_pairs)
    
    split_index = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]
    
    print(f"แบ่งข้อมูลเป็น: Train = {len(train_pairs)} คู่ | Val = {len(val_pairs)} คู่")

    # 6. คัดลอกไฟล์ไปยังโฟลเดอร์เป้าหมาย
    def copy_files(pairs, dest_img, dest_lbl):
        for img_name, lbl_name in pairs:
            shutil.copy(os.path.join(source_img_dir, img_name), os.path.join(dest_img, img_name))
            shutil.copy(os.path.join(source_lbl_dir, lbl_name), os.path.join(dest_lbl, lbl_name))

    print("กำลังจัดเตรียมและจัดส่งไฟล์ย่อยไปยังโฟลเดอร์ปลายทาง...")
    copy_files(train_pairs, train_img_path, train_lbl_path)
    copy_files(val_pairs, val_img_path, val_lbl_path)

    # 7. สร้างไฟล์ config.yaml
    # สามารถปรับแต่งชื่อคลาสตรงนี้ได้หากต้องการให้สื่อความหมาย
    class_names = {cid: f"class_{cid}" for cid in detected_classes}
    # หากต้องการกำหนดชื่อคลาสแบบระบุเองสามารถใส่เป็น map ตรงนี้ได้ เช่น:
    # class_names = {0: "rock", 1: "paper", 2: "scissors"}

    config_data = {
        'path': '/content/yolo_dataset',  # พาธปลายทางเมื่อ unzip ไปวางบน Google Colab
        'train': 'train/images',
        'val': 'val/images',
        'names': class_names
    }

    config_file_path = os.path.join(dataset_dir, "config.yaml")
    with open(config_file_path, 'w', encoding='utf-8') as cf:
        yaml.dump(config_data, cf, default_flow_style=False, sort_keys=True)
    print("สร้างไฟล์ 'config.yaml' เรียบร้อย")

    # 8. บีบอัด yolo_dataset เป็นไฟล์ data.zip
    zip_filename = "data.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    print(f"กำลังบีบอัดข้อมูลทั้งหมดเป็น '{zip_filename}'...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_full_path = os.path.join(root, file)
                # เก็บเฉพาะโครงสร้างย่อยภายใต้ yolo_dataset
                arcname = os.path.relpath(file_full_path, start=os.path.dirname(dataset_dir))
                zipf.write(file_full_path, arcname)

    print(f"เสร็จสมบูรณ์! คุณได้ไฟล์ '{zip_filename}' แล้ว พร้อมสำหรับอัปโหลดขึ้น Google Colab 🎉")

if __name__ == "__main__":
    main()
