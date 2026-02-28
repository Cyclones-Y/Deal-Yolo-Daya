import os
import json
import math
import random
import re
import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import requests
import pandas as pd
import yaml
from PIL import Image
import io
import logging

def safe_dataframe(df: pd.DataFrame):
    """
    Safely convert dataframe cells to string if needed to avoid Streamlit arrow errors.
    Returns a copy of the dataframe.
    """
    def _coerce_cell(value):
        if value is None:
            return ""
        try:
            if isinstance(value, float) and math.isnan(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return value.hex()
        return str(value)

    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].apply(_coerce_cell)
    return safe_df

def download_image(url: str, save_path: str) -> bool:
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"\n❌ 图片下载失败 {url}：{e}")
        return False

def json_to_yolo_annotation(
        json_str: str,
        img_width: Optional[float],
        img_height: Optional[float],
        class_mapping: Dict[str, int],
        class_id_counter: int
) -> Tuple[str, int, Dict[str, int]]:
    yolo_lines = []
    current_class_id = class_id_counter
    
    try:
        if pd.isna(json_str) or not isinstance(json_str, str):
            return "", current_class_id, class_mapping
        data = json.loads(json_str)
        objects = data.get("objects", [])
        img_width = img_width or data.get("width", 1)
        img_height = img_height or data.get("height", 1)

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            # 处理类别ID
            obj_name = obj.get("name", "unknown")
            if obj_name not in class_mapping:
                class_mapping[obj_name] = current_class_id
                current_class_id += 1
            cid = class_mapping[obj_name]

            # 处理坐标
            ptlist = obj.get("polygon", {}).get("ptList", [])
            if len(ptlist) != 2:
                continue
            p1, p2 = ptlist
            x1, y1 = min(p1["x"], p2["x"]), min(p1["y"], p2["y"])
            x2, y2 = max(p1["x"], p2["x"]), max(p1["y"], p2["y"])

            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # 限制范围在0~1之间
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999, width))
            height = max(0.001, min(0.999, height))

            yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"\n❌ 标注转换失败：{e}")

    return "\n".join(yolo_lines), current_class_id, class_mapping

def format_bytes(value):
    if value is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"

def format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_int(value):
    return "-" if value is None else f"{value:,}"

def format_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{(numerator / denominator) * 100:.1f}%"

def check_requirements():
    req_path = Path(__file__).resolve().parent.parent.parent.parent / "requirements.txt"
    if not req_path.exists():
        # Try finding it in project root if relative path fails
        req_path = Path("requirements.txt")
        if not req_path.exists():
            return ["requirements.txt 未找到"]
    
    mapping = {
        "Pillow": "PIL",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
    }
    missing = []
    try:
        for line in req_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = line.split("==")[0].strip()
            module = mapping.get(pkg, pkg)
            if importlib.util.find_spec(module) is None:
                missing.append(pkg)
    except Exception:
        pass
    return missing

def get_csv_columns(file_obj_or_path):
    try:
        if hasattr(file_obj_or_path, "getbuffer"):
            data = io.BytesIO(file_obj_or_path.getbuffer())
            df = pd.read_csv(data, nrows=1, encoding="utf-8-sig")
        else:
            df = pd.read_csv(file_obj_or_path, nrows=1, encoding="utf-8-sig")
        return list(df.columns)
    except Exception:
        return None

def get_row_count(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        path_lower = str(p).lower()
        if path_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
            return len(df)
        if path_lower.endswith(".csv"):
            line_count = 0
            with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
                for _ in f:
                    line_count += 1
            return max(line_count - 1, 0)
        df = pd.read_csv(p, encoding="utf-8-sig")
        return len(df)
    except Exception:
        return None

def get_image_count(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return len([f for f in p.iterdir() if f.is_file()])
    except Exception:
        return None

def list_excel_files(folder_path):
    if not folder_path:
        return []
    folder = Path(folder_path)
    if not folder.exists():
        return []
    files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    return sorted(files)

def list_subdirectories(path_str: str, include_hidden: bool = False, max_items: int = 200):
    if not path_str:
        return []
    base = Path(path_str)
    if not base.exists() or not base.is_dir():
        return []
    items = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if not include_hidden and p.name.startswith("."):
            continue
        items.append(p)
    items = sorted(items, key=lambda x: x.name.lower())
    return items[:max_items]

def list_yaml_files(path_str: str, max_items: int = 300):
    if not path_str:
        return []
    base = Path(path_str)
    if not base.exists():
        return []
    if base.is_file() and base.suffix.lower() in {".yaml", ".yml"}:
        return [base]
    patterns = ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]
    files = []
    for pattern in patterns:
        files.extend(base.rglob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file()})
    return files[:max_items]

def load_dataset_yaml(path_str: str):
    if not path_str:
        return None, "路径为空"
    path = Path(path_str)
    if not path.exists():
        return None, "未找到数据集配置文件"
    try:
        import yaml
    except Exception:
        return None, "未安装 pyyaml，无法读取数据集详情"
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return data, None
    except Exception as exc:
        return None, f"读取失败：{exc}"

def count_images_in_dir(dir_path: Path):
    if not dir_path or not dir_path.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    try:
        return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return None

def summarize_dataset(path_str: str):
    data, err = load_dataset_yaml(path_str)
    if err:
        return {"error": err}
    base_path = Path(path_str).parent
    root_value = data.get("path")
    if root_value:
        root_value = Path(root_value)
        root_path = root_value if root_value.is_absolute() else (base_path / root_value).resolve()
    else:
        root_path = base_path
    train_dir = root_path / str(data.get("train", ""))
    val_dir = root_path / str(data.get("val", ""))
    test_dir = root_path / str(data.get("test", ""))
    return {
        "nc": data.get("nc"),
        "names": data.get("names"),
        "path": str(root_path),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
        "train_images": count_images_in_dir(train_dir),
        "val_images": count_images_in_dir(val_dir),
        "test_images": count_images_in_dir(test_dir),
    }

def get_cuda_summary():
    try:
        import torch
    except Exception:
        return {"available": False, "detail": "未安装 torch"}
    available = torch.cuda.is_available()
    if not available:
        return {"available": False, "detail": "CUDA 不可用"}
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"available": True, "detail": f"{len(devices)} 张GPU", "devices": devices}

def list_dataset_roots_from_configs(config_paths):
    roots = []
    for path in config_paths:
        try:
            data, err = load_dataset_yaml(str(path))
            if err or not data:
                roots.append(Path(path).parent.resolve())
                continue
            base_path = Path(path).parent
            root_value = data.get("path")
            if root_value:
                root_value = Path(root_value)
                root_path = root_value if root_value.is_absolute() else (base_path / root_value).resolve()
            else:
                root_path = base_path
            roots.append(root_path)
        except Exception:
            roots.append(Path(path).parent.resolve())
    unique = []
    seen = set()
    for item in roots:
        if str(item) not in seen:
            unique.append(item)
            seen.add(str(item))
    return unique

def collect_image_files(dir_path: Path, max_images: int = 24, shuffle: bool = True, recursive: bool = True):
    if not dir_path or not dir_path.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return []
    if shuffle:
        random.shuffle(files)
    return files[:max_images]

def scan_dataset_configs(root_str: str):
    root = Path(root_str) if root_str else None
    if not root or not root.exists():
        return []
    patterns = ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]
    found = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    unique = sorted({p.resolve() for p in found if p.is_file()})
    return unique

def get_dir_stats(path: Path, recursive: bool = False, max_files: int = 5000, max_depth: int = 6):
    if not path.exists() or not path.is_dir():
        return {"files": 0, "dirs": 0, "bytes": 0, "truncated": False}
    total_bytes = 0
    files = 0
    dirs = 0
    truncated = False

    if not recursive:
        for item in path.iterdir():
            if item.is_dir():
                dirs += 1
            elif item.is_file():
                files += 1
                try:
                    total_bytes += item.stat().st_size
                except Exception:
                    pass
        return {"files": files, "dirs": dirs, "bytes": total_bytes, "truncated": False}

    base_depth = len(path.parts)
    for root, dirnames, filenames in os.walk(path):
        depth = len(Path(root).parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirs += len(dirnames)
        for name in filenames:
            files += 1
            if files > max_files:
                truncated = True
                return {"files": files, "dirs": dirs, "bytes": total_bytes, "truncated": True}
            file_path = Path(root) / name
            try:
                total_bytes += file_path.stat().st_size
            except Exception:
                pass
    return {"files": files, "dirs": dirs, "bytes": total_bytes, "truncated": truncated}

def list_image_files_for_preview(path_str: str, recursive: bool, max_files: int):
    base = Path(path_str)
    if not base.exists() or not base.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    if recursive:
        for root, _, filenames in os.walk(base):
            for name in filenames:
                file_path = Path(root) / name
                if file_path.suffix.lower() in exts:
                    try:
                        stat = file_path.stat()
                        files.append({"path": str(file_path), "size": stat.st_size, "mtime": stat.st_mtime})
                    except Exception:
                        files.append({"path": str(file_path), "size": 0, "mtime": 0})
                    if len(files) >= max_files:
                        return files
    else:
        for p in base.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    stat = p.stat()
                    files.append({"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime})
                except Exception:
                    files.append({"path": str(p), "size": 0, "mtime": 0})
                if len(files) >= max_files:
                    break
    return files

def get_immediate_children_sizes(path: Path, max_items: int = 10):
    if not path.exists() or not path.is_dir():
        return [], []
    dir_sizes = []
    file_sizes = []
    for entry in path.iterdir():
        if entry.is_dir():
            size = 0
            try:
                for item in entry.iterdir():
                    if item.is_file():
                        try:
                            size += item.stat().st_size
                        except Exception:
                            pass
            except Exception:
                pass
            dir_sizes.append((entry.name, size))
        elif entry.is_file():
            try:
                size = entry.stat().st_size
            except Exception:
                size = 0
            file_sizes.append((entry.name, size))
    dir_sizes = sorted(dir_sizes, key=lambda x: x[1], reverse=True)[:max_items]
    file_sizes = sorted(file_sizes, key=lambda x: x[1], reverse=True)[:max_items]
    return dir_sizes, file_sizes

def collect_dir_paths(root: Path, include_hidden: bool, max_depth: int, max_nodes: int):
    paths = []

    def _walk(path: Path, depth: int):
        if depth > max_depth or len(paths) >= max_nodes:
            return
        try:
            children = [
                p for p in path.iterdir()
                if p.is_dir() and (include_hidden or not p.name.startswith("."))
            ]
        except Exception:
            return
        for child in sorted(children, key=lambda x: x.name.lower()):
            if len(paths) >= max_nodes:
                break
            paths.append(str(child))
            _walk(child, depth + 1)

    _walk(root, 1)
    return paths

def list_immediate_dirs(path_str: str, include_hidden: bool = False):
    base = Path(path_str)
    if not base.exists() or not base.is_dir():
        return []
    items = [
        p for p in base.iterdir()
        if p.is_dir() and (include_hidden or not p.name.startswith("."))
    ]
    return sorted(items, key=lambda x: x.name.lower())

def get_path_suggestions(current_value: str, include_hidden: bool = False, max_items: int = 50):
    if not current_value:
        return []
    expanded = os.path.expanduser(current_value)
    candidate = Path(expanded)
    parent = candidate if candidate.is_dir() else candidate.parent
    if not parent.exists():
        return []
    items = list_immediate_dirs(str(parent), include_hidden=include_hidden)
    suggestions = [str(p) for p in items]
    if current_value not in suggestions:
        suggestions.insert(0, current_value)
    return suggestions[:max_items]

def search_directories(root: Path, query: str, include_hidden: bool, max_results: int = 60):
    if not root.exists() or not query:
        return []
    query_lower = query.lower()
    results = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if not include_hidden and path.name.startswith("."):
            continue
        if query_lower in path.name.lower():
            results.append(path)
            if len(results) >= max_results:
                break
    return results

def safe_filename(value: str) -> str:
    if not value:
        return "train"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "train"

def parse_kv_lines(text: str):
    options = {}
    errors = []
    if not text:
        return options, errors
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"无法解析：{raw_line}")
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            errors.append(f"参数名为空：{raw_line}")
            continue
        value = raw_value
        try:
            value = json.loads(raw_value)
        except Exception:
            lowered = raw_value.lower()
            if lowered in {"true", "false"}:
                value = lowered == "true"
            elif lowered in {"none", "null"}:
                value = None
            else:
                try:
                    if "." in raw_value:
                        value = float(raw_value)
                    else:
                        value = int(raw_value)
                except Exception:
                    value = raw_value
        options[key] = value
    return options, errors

def save_upload(uploaded_file, dest_path: Path):
    """
    上传文件保存方法，增加异常捕获、校验和日志
    """
    logger = logging.getLogger(__name__)
    # 前置校验：上传文件是否有效
    if uploaded_file is None:
        logger.error("上传文件为空，无法保存")
        raise ValueError("上传文件不能为空")

    # 校验文件大小
    file_size = uploaded_file.size
    if file_size == 0:
        logger.error(f"上传文件 {uploaded_file.name} 为空文件（大小：0字节）")
        raise ValueError(f"上传文件 {uploaded_file.name} 为空")

    # 创建目录（带权限检查）
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)  # Linux/macOS 权限
    except PermissionError as e:
        logger.error(f"创建目录失败：{dest_path.parent}，权限不足：{e}")
        raise PermissionError(f"无写入权限：{dest_path.parent}") from e

    # 写入文件（带校验）
    try:
        with open(dest_path, "wb") as f:
            # 分块写入大文件，避免缓冲区溢出
            chunk_size = 1024 * 1024  # 1MB 分块
            buffer = uploaded_file.getbuffer()
            f.write(buffer)

        # 校验写入后文件大小
        saved_size = dest_path.stat().st_size
        if saved_size != file_size:
            logger.warning(
                f"文件 {uploaded_file.name} 写入不完整！原大小：{file_size} 字节，保存后：{saved_size} 字节"
            )
            raise RuntimeError(f"文件写入不完整，丢失 {file_size - saved_size} 字节数据")

        logger.info(f"文件 {uploaded_file.name} 保存成功，路径：{dest_path}，大小：{saved_size} 字节")
        return dest_path

    except Exception as e:
        logger.error(f"保存文件失败：{e}", exc_info=True)
        # 清理不完整文件
        if dest_path.exists():
            dest_path.unlink()
        raise

def save_uploads(uploaded_files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in uploaded_files:
        out_path = dest_dir / item.name
        save_upload(item, out_path)
        paths.append(out_path)
    return paths

def _safe_filename(value: str) -> str:
    return safe_filename(value)

def _safe_dataset_dir_name(category_name, default_name):
    if not category_name:
        return default_name
    return safe_filename(str(category_name))

def _split_label_cell(cell_value):
    if pd.isna(cell_value):
        return []
    text = str(cell_value).strip()
    if not text:
        return []
    # Split by common separators
    tokens = re.split(r"[,，;；|]", text)
    return [t.strip() for t in tokens if t.strip()]

def _parse_data_objects(json_str):
    if pd.isna(json_str) or not isinstance(json_str, str) or not json_str:
        return None, [], "空数据"
    try:
        data = json.loads(json_str)
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            return data, [], "objects不是列表"
        return data, objects, None
    except json.JSONDecodeError:
        return None, [], "JSON解析失败"
    except Exception as e:
        return None, [], str(e)

def _split_object_labels(raw_name):
    if not raw_name:
        return []
    return [t.strip() for t in re.split(r"[,，;；|]", str(raw_name)) if t.strip()]

def _replace_label_tokens(raw_name, label_map):
    if not raw_name:
        return raw_name, 0, 0
    tokens = _split_object_labels(raw_name)
    new_tokens = []
    replaced_count = 0
    for t in tokens:
        if t in label_map:
            new_tokens.append(label_map[t])
            replaced_count += 1
        else:
            new_tokens.append(t)
    # Deduplicate and sort to ensure deterministic order
    unique_tokens = sorted(list(set(new_tokens)))
    new_name = ",".join(unique_tokens)
    return new_name, replaced_count, len(tokens)

def _extract_boxes_with_labels(json_str):
    boxes = [] # (label, x1, y1, x2, y2)
    try:
        if pd.isna(json_str) or not isinstance(json_str, str):
            return boxes
        data = json.loads(json_str)
        objects = data.get("objects", [])
        for obj in objects:
            if not isinstance(obj, dict): continue
            label = obj.get("name")
            if not label: continue
            
            # Check for ptList (polygon) first, then try to find box
            # Ideally we expect BBox format from previous steps (replace_ptlist)
            # If still polygon, we convert it here implicitly or just skip?
            # The generate_yolo function expects boxes.
            
            # Let's support both if possible, but mainly ptList -> box
            ptlist = obj.get("polygon", {}).get("ptList", [])
            if not ptlist: continue
            
            xs = [p.get("x") for p in ptlist if isinstance(p, dict) and "x" in p]
            ys = [p.get("y") for p in ptlist if isinstance(p, dict) and "y" in p]
            if not xs or not ys: continue
            
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            boxes.append((label, x1, y1, x2, y2))
    except: pass
    return boxes

def _safe_image_stem(source_url, idx):
    if not source_url:
        return f"img_{idx}"
    try:
        name = Path(str(source_url)).name
        stem = Path(name).stem
        # Remove query params if any
        if "?" in stem:
            stem = stem.split("?")[0]
        safe = safe_filename(stem)
        return f"{safe}_{idx}"
    except:
        return f"img_{idx}"

def _ensure_image_cached(source_url, cache_dir: Path):
    if not source_url: return None
    try:
        # Check if source is local path
        if Path(source_url).exists():
             return Path(source_url)
        
        # It's a URL
        filename = source_url.split("/")[-1]
        if "?" in filename:
            filename = filename.split("?")[0]
        if not filename:
            filename = f"image_{hash(source_url)}.jpg"
            
        cache_path = cache_dir / filename
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path
            
        download_image(source_url, str(cache_path))
        if cache_path.exists():
            return cache_path
    except: pass
    return None
