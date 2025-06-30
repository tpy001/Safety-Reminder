import os 
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


from huggingface_hub import snapshot_download
from huggingface_hub import login
login("Your hugginface access token here.")



# 仓库ID (作者/仓库名)
repo_id = "sqrti/SPA-VL"
# 指定你想将文件下载到哪个本地文件夹
local_dir = "./data/SPA_VL"

if os.exists(local_dir):
    print("Dataset has been downloaded to the local directory.")
    exit(0)
    
print(f"正在将仓库 {repo_id} 的所有文件下载到 {local_dir}...")

# snapshot_download 会下载指定仓库的所有文件
# 它也会使用缓存，并支持断点续传
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",  # 明确这是一个数据集类型的仓库
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 建议设为False，会直接复制文件而非创建符号链接
)

print(f"\n下载完成！所有文件都已保存在 '{local_dir}' 文件夹中。")