# datasets

数据集配置目录。仓库仅保留 `coco128.yaml` 配置，图片数据不入库（见根 `.gitignore` 的 `datasets/coco128/`）。

- `coco128.yaml`：COCO128 本地数据集定义，`path` 指向 `datasets/coco128`（相对仓库根目录解析）。
- 预期目录结构：

  ```
  datasets/coco128/
  ├─ images/train2017/   # 128 张图片
  └─ labels/train2017/   # 对应的 YOLO 格式标注
  ```

准备数据：将 COCO128（Ultralytics 官方资源 `coco128.zip`）解压到 `datasets/coco128` 即可。

在仓库根目录执行训练：

```powershell
python training/train.py
```
