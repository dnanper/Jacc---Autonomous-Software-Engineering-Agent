from datasets import load_dataset
import pandas as pd
import json

def process_swe_bench_lite():
    # 1. Load dataset từ Hugging Face
    print("Đang tải dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    # 2. Chuyển sang Pandas DataFrame để dễ thao tác
    df = dataset.to_pandas()
    
    # 3. Xử lý cột thời gian 'created_at'
    # Cần đảm bảo cột này ở dạng datetime để sort chính xác
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # 4. Sắp xếp toàn bộ dữ liệu:
    # Ưu tiên 1: Theo tên Repo
    # Ưu tiên 2: Theo thời gian tạo (cũ nhất -> mới nhất)
    df_sorted = df.sort_values(by=['repo', 'created_at'], ascending=[True, True])
    
    # 5. Gom nhóm (Grouping)
    # Kết quả sẽ là một dictionary: Key là tên Repo, Value là danh sách các task
    grouped_tasks = {}
    
    # Lấy danh sách các repo duy nhất
    unique_repos = df_sorted['repo'].unique()
    
    print(f"Đã tìm thấy {len(unique_repos)} repositories.")
    
    for repo in unique_repos:
        # Lấy tất cả row thuộc repo đó (đã được sort theo time ở bước 4)
        repo_tasks = df_sorted[df_sorted['repo'] == repo]
        
        # Chuyển về dạng list các dictionary (records) để dễ dùng sau này
        # Cần convert timestamp sang string để serialize JSON nếu cần
        tasks_list = json.loads(repo_tasks.to_json(orient="records", date_format="iso"))
        
        grouped_tasks[repo] = tasks_list
        print(f" -> Repo '{repo}': {len(tasks_list)} tasks")

    return grouped_tasks

# --- Thực thi ---
if __name__ == "__main__":
    result = process_swe_bench_lite()
    
    # (Tùy chọn) Lưu ra file JSON để dùng cho Agent Jacc
    output_file = "swe_bench_lite_grouped.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nĐã lưu kết quả vào {output_file}")
    
    # Ví dụ truy cập thử:
    first_repo = list(result.keys())[0]
    print(f"\nVí dụ task đầu tiên của repo {first_repo}:")
    print(f"ID: {result[first_repo][0]['instance_id']}")
    print(f"Created At: {result[first_repo][0]['created_at']}")