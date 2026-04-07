import os
import glob

def find_top_cd_files(polars_dir, top_n=10):
    polar_files = glob.glob(os.path.join(polars_dir, "*_polar.txt"))
    file_max_cds = []

    print(f"正在扫描 {len(polar_files)} 个极曲线文件...")

    for p_file in polar_files:
        max_cd_in_file = -1.0
        
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(p_file, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        start_idx = -1
        for i, line in enumerate(lines):
            if '------' in line:
                start_idx = i + 1
                break
        
        if start_idx == -1:
            continue

        for line in lines[start_idx:]:
            if not line.strip():
                continue
            vals = line.split()
            if len(vals) < 3:
                continue
            
            try:
                cd = float(vals[2])
                if cd > max_cd_in_file:
                    max_cd_in_file = cd
            except ValueError:
                continue
        
        if max_cd_in_file != -1.0:
            file_max_cds.append((os.path.basename(p_file), max_cd_in_file))

    # 按 Cd 降序排序
    file_max_cds.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCd 最大的前 {top_n} 个文件:")
    for i, (fname, max_cd) in enumerate(file_max_cds[:top_n], 1):
        print(f"{i}. {fname} (Max Cd: {max_cd:.6f})")

if __name__ == '__main__':
    polars_path = os.path.join("foildata","polars")
    find_top_cd_files(polars_path)
