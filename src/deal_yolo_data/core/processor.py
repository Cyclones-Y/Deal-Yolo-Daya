import os
import json
import copy
import re
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
import requests

from .utils import (
    _safe_filename,
    _safe_dataset_dir_name,
    _split_label_cell,
    _parse_data_objects,
    _split_object_labels,
    _replace_label_tokens,
    _extract_boxes_with_labels,
    _safe_image_stem,
    _ensure_image_cached,
    save_upload
)

def merge_all_csv_in_folder(
        folder_path,
        output_file="merged_csv.csv",
        encoding="utf-8-sig",
        chunk_size: int = 100000,
        progress_callback=None,
):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在：{folder_path}")

    csv_files = list(Path(folder_path).glob("*.csv"))
    if not csv_files:
        print(f"警告：文件夹 {folder_path} 中未找到CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个CSV文件，开始合并...")

    output_file = str(output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    total_rows = 0
    total_bytes = sum(f.stat().st_size for f in csv_files)
    completed_bytes = 0

    for file_idx, csv_file in enumerate(csv_files, start=1):
        try:
            file_size = csv_file.stat().st_size
            if progress_callback:
                progress_callback(
                    file_idx,
                    len(csv_files),
                    csv_file.name,
                    total_rows,
                    0,
                    0,
                    file_size,
                    0,
                    total_bytes,
                    completed_bytes,
                )
            file_rows = 0
            with open(csv_file, "r", encoding=encoding, errors="ignore") as f:
                chunk_iter = pd.read_csv(
                    f,
                    parse_dates=False,
                    chunksize=chunk_size,
                )
                for chunk_idx, df in enumerate(chunk_iter, start=1):
                    df["source_file"] = os.path.basename(csv_file)
                    mode = "w" if not header_written else "a"
                    header = not header_written
                    df.to_csv(output_file, index=False, encoding=encoding, mode=mode, header=header)
                    header_written = True
                    rows = len(df)
                    file_rows += rows
                    total_rows += rows
                    file_bytes = f.tell()
                    total_bytes_read = completed_bytes + file_bytes
                    if progress_callback:
                        progress_callback(
                            file_idx,
                            len(csv_files),
                            csv_file.name,
                            total_rows,
                            file_rows,
                            chunk_idx,
                            file_size,
                            file_bytes,
                            total_bytes,
                            total_bytes_read,
                        )
            print(f"成功读取：{csv_file.name}（{file_rows}行）")
            completed_bytes += file_size
        except Exception as e:
            print(f"读取失败 {csv_file.name}：{str(e)}")
            continue

    if not header_written:
        print("错误：没有可合并的有效CSV数据")
        return None

    print(f"\n合并完成！共 {total_rows} 行数据")
    print(f"输出文件：{os.path.abspath(output_file)}")
    return total_rows

def deduplicate_csv_by_source(
        csv_path: str,
        output_file: Optional[str] = "deduplicate_result.csv",
        encoding: str = "utf-8-sig",
        keep: str = "first",
        verbose: bool = True
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

    if not csv_path.endswith(".csv"):
        raise ValueError(f"文件不是CSV格式：{csv_path}（请传入.csv后缀的文件）")

    try:
        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            parse_dates=False
        )
        if verbose:
            print(f"成功读取CSV文件：{os.path.basename(csv_path)}")
            print(f"读取后原始数据行数：{len(df)}")
    except Exception as e:
        raise Exception(f"读取CSV文件失败：{str(e)}") from e

    if "source" not in df.columns:
        raise KeyError(f"CSV文件中未找到'source'列，请检查列名是否正确（当前列名：{list(df.columns)}）")

    original_count = len(df)
    deduplicated_df = df.drop_duplicates(
        subset=["source"],
        keep=keep,
        ignore_index=True
    )
    duplicate_count = original_count - len(deduplicated_df)

    if verbose:
        print(f"去重策略：按'source'列保留{keep}条数据")
        print(f"去除重复数据行数：{duplicate_count}")
        print(f"去重后剩余数据行数：{len(deduplicated_df)}")

    if output_file is not None:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            deduplicated_df.to_csv(output_file, index=False, encoding=encoding)
            if verbose:
                print(f"去重后的文件已保存至：{os.path.abspath(output_file)}")
        except Exception as e:
            raise Exception(f"保存去重文件失败：{str(e)}") from e

    return deduplicated_df

def remove_duplicates_between_csv(
        main_csv: str,
        ref_csv: str,
        output_csv: str = "filtered_main.csv",
        compare_col: str = "source",
        encoding: str = "utf-8-sig",
        verbose: bool = True
) -> pd.DataFrame:
    for csv_path in [main_csv, ref_csv]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在：{csv_path}")
        if not csv_path.endswith(".csv"):
            raise ValueError(f"文件不是CSV格式：{csv_path}（请传入.csv后缀文件）")

    try:
        df_main = pd.read_csv(main_csv, encoding=encoding, parse_dates=False)
        df_ref = pd.read_csv(ref_csv, encoding=encoding, parse_dates=False)
        if verbose:
            print(f"读取主文件：{len(df_main)}行")
            print(f"读取参考文件：{len(df_ref)}行")
    except Exception as e:
        raise Exception(f"读取CSV失败：{str(e)}") from e

    if compare_col not in df_main.columns:
        raise KeyError(f"主文件中未找到列 '{compare_col}'")
    if compare_col not in df_ref.columns:
        raise KeyError(f"参考文件中未找到列 '{compare_col}'")

    ref_values = set(df_ref[compare_col].dropna().astype(str))
    
    # 筛选出不在参考文件中的记录
    # 注意：这里需要处理 NaN 值和类型转换
    is_duplicate = df_main[compare_col].astype(str).isin(ref_values)
    df_filtered = df_main[~is_duplicate].copy()
    
    removed_count = len(df_main) - len(df_filtered)

    if verbose:
        print(f"去重依据列：{compare_col}")
        print(f"参考文件中唯一值数量：{len(ref_values)}")
        print(f"剔除重复行数：{removed_count}")
        print(f"保留行数：{len(df_filtered)}")

    try:
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df_filtered.to_csv(output_csv, index=False, encoding=encoding)
        if verbose:
            print(f"结果已保存至：{os.path.abspath(output_csv)}")
    except Exception as e:
        raise Exception(f"保存结果失败：{str(e)}") from e

    return df_filtered

def overwrite_reference_with_result(result_csv: str, ref_csv: str):
    if not os.path.exists(result_csv):
        raise FileNotFoundError(f"结果文件不存在：{result_csv}")
    
    # 如果参考文件不存在，直接复制
    import shutil
    shutil.copy2(result_csv, ref_csv)

def process_csv_replace_ptlist(
        input_csv_path: str,
        output_csv_path: str = "processed_replaced_ptlist.csv",
        excluded_output_file: Optional[str] = "processed_excluded.csv"
):
    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
        print(f"成功读取CSV，共 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_csv_path}")
        return None
    except Exception as e:
        print(f"读取失败：{e}")
        return None

    required_col = "结果字段-目标检测标签配置"
    if required_col not in df.columns:
        print(f"错误：CSV缺少列 '{required_col}'")
        return None

    filtered_df = df.dropna(subset=[required_col]).copy()
    excluded_df = df[df[required_col].isna()].copy()

    def get_bbox_points(ptlist):
        valid_points = [p for p in ptlist if isinstance(p, dict) and "x" in p and "y" in p]
        if not valid_points:
            return [{"x": None, "y": None}, {"x": None, "y": None}]
        min_x = min(p["x"] for p in valid_points)
        max_x = max(p["x"] for p in valid_points)
        min_y = min(p["y"] for p in valid_points)
        max_y = max(p["y"] for p in valid_points)
        return [{"x": min_x, "y": min_y}, {"x": max_x, "y": max_y}]

    def parse_and_replace_ptlist(json_str):
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return None
            data = json.loads(json_str)
            objects = data.get("objects", [])
            updated_objects = []
            for obj in objects:
                if isinstance(obj, dict):
                    updated_obj = obj.copy()
                    original_ptlist = obj.get("polygon", {}).get("ptList", [])
                    updated_ptlist = get_bbox_points(original_ptlist)
                    if "polygon" not in updated_obj:
                        updated_obj["polygon"] = {}
                    updated_obj["polygon"]["ptList"] = updated_ptlist
                    updated_objects.append(updated_obj)
            data["objects"] = updated_objects
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            return None

    filtered_df["新_结果字段-目标检测标签配置"] = filtered_df["结果字段-目标检测标签配置"].apply(parse_and_replace_ptlist)

    def extract_width_height(json_str):
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return {"width": None, "height": None}
            data = json.loads(json_str)
            return {"width": data.get("width"), "height": data.get("height")}
        except:
            return {"width": None, "height": None}

    wh_data = filtered_df["结果字段-目标检测标签配置"].apply(extract_width_height)
    filtered_df["width"] = [item["width"] for item in wh_data]
    filtered_df["height"] = [item["height"] for item in wh_data]

    result_cols = [
        "source",
        "结果字段-目标检测标签配置",
        "新_结果字段-目标检测标签配置",
        "width", "height"
    ]
    
    # 保留其他可能存在的列
    existing_cols = [c for c in result_cols if c in filtered_df.columns]
    
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    filtered_df[existing_cols].to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    if excluded_output_file is not None:
        Path(excluded_output_file).parent.mkdir(parents=True, exist_ok=True)
        excluded_df.to_csv(excluded_output_file, index=False, encoding="utf-8-sig")

    return {
        "filtered_rows": len(filtered_df),
        "excluded_rows": len(excluded_df),
        "excluded_output": excluded_output_file,
    }

def filter_by_box_count_and_iou(
        input_csv_path,
        high_iou_csv="high_iou_0.98.csv",
        other_csv="other_data.csv",
        min_boxes: int = 2,
        iou_threshold: float = 0.98
):
    def calculate_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        intersection = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if intersection == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union != 0 else 0.0

    def extract_boxes(json_str):
        boxes = []
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return boxes
            data = json.loads(json_str)
            objects = data.get("objects", [])
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                ptlist = obj.get("polygon", {}).get("ptList", [])
                if len(ptlist) != 2:
                    continue
                p1, p2 = ptlist
                if not (isinstance(p1, dict) and isinstance(p2, dict)
                        and "x" in p1 and "y" in p1
                        and "x" in p2 and "y" in p2):
                    continue
                x1 = min(p1["x"], p2["x"])
                y1 = min(p1["y"], p2["y"])
                x2 = max(p1["x"], p2["x"])
                y2 = max(p1["y"], p2["y"])
                boxes.append((x1, y1, x2, y2))
        except Exception:
            pass
        return boxes

    def meet_conditions(boxes):
        if len(boxes) < min_boxes:
            return False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = calculate_iou(boxes[i], boxes[j])
                if iou >= iou_threshold:
                    return True
        return False

    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取失败：{e}")
        return

    required_col = "新_结果字段-目标检测标签配置"
    if required_col not in df.columns:
        print(f"错误：缺少必要列 {required_col}")
        return

    high_iou_data = []
    other_data = []

    for idx, row in df.iterrows():
        json_str = row[required_col]
        boxes = extract_boxes(json_str)
        if meet_conditions(boxes):
            high_iou_data.append(row)
        else:
            other_data.append(row)

    Path(high_iou_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(other_csv).parent.mkdir(parents=True, exist_ok=True)

    high_iou_df = pd.DataFrame(high_iou_data, columns=df.columns)
    high_iou_df.to_csv(high_iou_csv, index=False, encoding="utf-8-sig")

    other_df = pd.DataFrame(other_data, columns=df.columns)
    other_df.to_csv(other_csv, index=False, encoding="utf-8-sig")

def download_and_draw_annotations(
        input_csv_path,
        output_dir: Optional[str] = None,
        download_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
        max_images: Optional[int] = None,
        timeout: int = 15
):
    base_dir = Path(output_dir) if output_dir else Path(os.getcwd())
    download_dir = Path(download_dir) if download_dir else (base_dir / "downloaded_images")
    result_dir = Path(result_dir) if result_dir else (base_dir / "annotated_images")
    download_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取CSV失败：{e}")
        return

    required_cols = ["source", "结果字段-目标检测标签配置", "新_结果字段-目标检测标签配置"]
    if any(col not in df.columns for col in required_cols):
        print("CSV缺少必要列")
        return

    def get_font():
        try:
            return ImageFont.truetype("simhei.ttf", 48)
        except:
            try:
                return ImageFont.truetype("Arial Unicode.ttf", 48)
            except:
                return ImageFont.load_default()

    font = get_font()

    def draw_annotation_boxes(image, json_str, color, draw):
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return
            data = json.loads(json_str)
            objects = data.get("objects", [])
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                name = obj.get("name", "未知类别")
                ptlist = obj.get("polygon", {}).get("ptList", [])
                points = []
                for p in ptlist:
                    if (isinstance(p, dict) and "x" in p and "y" in p
                            and p["x"] is not None and p["y"] is not None):
                        points.append((p["x"], p["y"]))
                if len(points) < 2:
                    continue
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    text_bbox = draw.textbbox((x1, y1 - 20), name, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255, 180))
                    draw.text((x1, y1 - 20), name, font=font, fill=color)
                else:
                    draw.polygon(points, outline=color, width=2)
                    min_x = min(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    text_bbox = draw.textbbox((min_x, min_y - 20), name, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255, 180))
                    draw.text((min_x, min_y - 20), name, font=font, fill=color)
        except Exception:
            pass

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        processed_count = success_count + fail_count
        if max_images is not None and processed_count >= max_images:
            break
        source_url = row["source"]
        original_anno = row["结果字段-目标检测标签配置"]
        new_anno = row["新_结果字段-目标检测标签配置"]

        img_filename = source_url.split("/")[-1] if "/" in source_url else f"image_{idx}.jpg"
        download_path = download_dir / img_filename
        result_path = result_dir / img_filename

        if not os.path.exists(download_path):
            try:
                response = requests.get(source_url, stream=True, timeout=timeout)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception:
                fail_count += 1
                continue
        
        try:
            with Image.open(download_path) as img:
                draw = ImageDraw.Draw(img)
                draw_annotation_boxes(img, original_anno, (255, 0, 0), draw)
                draw_annotation_boxes(img, new_anno, (0, 255, 0), draw)
                img.save(result_path)
            success_count += 1
        except Exception:
            fail_count += 1

def replace_labels_by_mapping(
        input_csv_path: str,
        mapping_excel_path: str,
        output_csv_path: str,
        sheet_name: Optional[str] = None,
        old_col: Optional[str] = None,
        new_col: Optional[str] = None,
        json_columns: Optional[list] = None,
        diff_excel_path: Optional[str] = None,
        unmatched_excel_path: Optional[str] = None,
        sample_size: int = 30,
):
    # Imported from utils.py in app.py logic, but re-implemented here with imports
    # Using the logic provided in app.py
    df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    mapping_df = pd.read_excel(mapping_excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(mapping_excel_path)
    
    if not old_col or not new_col:
        cols = list(mapping_df.columns)
        if len(cols) < 2:
            raise ValueError("标签对照表至少需要两列")
        old_col = old_col or cols[0]
        new_col = new_col or cols[1]

    label_map = {}
    for _, row in mapping_df.iterrows():
        old_label = str(row.get(old_col, "")).strip()
        new_label = str(row.get(new_col, "")).strip()
        if old_label and old_label.lower() != "nan" and new_label and new_label.lower() != "nan":
            label_map[old_label] = new_label

    if json_columns is None:
        json_columns = []
        if "新_结果字段-目标检测标签配置" in df.columns:
            json_columns.append("新_结果字段-目标检测标签配置")
        if "结果字段-目标检测标签配置" in df.columns:
            json_columns.append("结果字段-目标检测标签配置")
    
    total_rows = len(df)
    total_objects = 0
    total_labels = 0
    replaced_labels = 0
    replaced_objects = 0
    replaced_rows = 0
    invalid_json_rows = 0
    missing_name_objects = 0
    unmatched_counter = {}
    diff_rows = []

    for idx, row in df.iterrows():
        row_replaced = False
        for col in json_columns:
            if col not in df.columns:
                continue
            json_str = row.get(col)
            if pd.isna(json_str) or not isinstance(json_str, str) or not json_str:
                continue
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                invalid_json_rows += 1
                continue
            objects = data.get("objects")
            if not isinstance(objects, list):
                continue
            row_diff = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                total_objects += 1
                raw_name = obj.get("name")
                if raw_name is None:
                    missing_name_objects += 1
                    continue
                labels = _split_object_labels(raw_name)
                for lbl in labels:
                    if lbl not in label_map:
                        unmatched_counter[lbl] = unmatched_counter.get(lbl, 0) + 1
                new_name, replaced, label_count = _replace_label_tokens(raw_name, label_map)
                total_labels += label_count
                if replaced > 0:
                    obj["name"] = new_name
                    replaced_labels += replaced
                    replaced_objects += 1
                    row_replaced = True
                if raw_name != new_name:
                    row_diff.append((raw_name, new_name))
            data["objects"] = objects
            df.at[idx, col] = json.dumps(data, ensure_ascii=False)
            if row_diff:
                diff_rows.append({
                    "source": row.get("source"),
                    "column": col,
                    "before": "；".join([p[0] for p in row_diff]),
                    "after": "；".join([p[1] for p in row_diff]),
                })
        if row_replaced:
            replaced_rows += 1

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    diff_path = None
    if diff_excel_path:
        diff_path = Path(diff_excel_path)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diff_rows).to_excel(diff_path, index=False)

    unmatched_path = None
    if unmatched_excel_path:
        unmatched_path = Path(unmatched_excel_path)
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        if unmatched_counter:
            pd.DataFrame([{"标签": k, "数量": v} for k, v in unmatched_counter.items()]).sort_values("数量", ascending=False).to_excel(unmatched_path, index=False)
        else:
            pd.DataFrame(columns=["标签", "数量"]).to_excel(unmatched_path, index=False)

    summary = {
        "total_rows": total_rows,
        "replaced_rows": replaced_rows,
        "total_objects": total_objects,
        "replaced_objects": replaced_objects,
        "total_labels": total_labels,
        "replaced_labels": replaced_labels,
        "invalid_json_rows": invalid_json_rows,
        "missing_name_objects": missing_name_objects,
        "mapping_size": len(label_map),
        "unmatched_labels": len(unmatched_counter),
    }
    return {
        "output_csv": output_csv_path,
        "summary": summary,
        "diff": diff_path,
        "unmatched": unmatched_path,
        "sample_diff": diff_rows[:sample_size],
    }

def split_dataset_by_rules(
        input_csv_path: str,
        rules_excel_path: str,
        output_dir: str,
        rule_mode: str = "wide",
        sheet_name: Optional[str] = None,
        label_col: Optional[str] = None,
        category_col: Optional[str] = None,
        json_columns: Optional[list] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
):
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"输入CSV不存在：{input_csv_path}")
    if not os.path.exists(rules_excel_path):
        raise FileNotFoundError(f"规则Excel不存在：{rules_excel_path}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

    if json_columns is None:
        json_columns = []
        if "新_结果字段-目标检测标签配置" in df.columns:
            json_columns.append("新_结果字段-目标检测标签配置")
        if "结果字段-目标检测标签配置" in df.columns:
            json_columns.append("结果字段-目标检测标签配置")
    
    rules_df = pd.read_excel(rules_excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(rules_excel_path)
    label_to_category = {}

    if rule_mode == "wide":
        for col in rules_df.columns:
            category = str(col).strip()
            if not category: continue
            for cell in rules_df[col].dropna():
                labels = _split_label_cell(cell)
                for label in labels:
                    label_to_category[label] = category
    elif rule_mode == "two_column":
        for _, row in rules_df.iterrows():
            label = str(row.get(label_col, "")).strip()
            category = str(row.get(category_col, "")).strip()
            if label and category and label.lower() != "nan" and category.lower() != "nan":
                label_to_category[label] = category

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_rows = {}
    unclassified_rows = []
    split_counts_rows = []

    for _, row in df.iterrows():
        json_str = None
        for col in json_columns:
            if col in row and isinstance(row[col], str) and row[col]:
                json_str = row[col]
                break

        data, objects, error = _parse_data_objects(json_str)
        if error or not objects:
            row_copy = row.copy()
            row_copy["无法分类原因"] = error or "标注字段objects为空"
            unclassified_rows.append(row_copy)
            split_counts_rows.append({
                "source": row.get("source"),
                "原始标签组合": "",
                "拆分条数": 0,
                "是否可分类": "否",
                "无法分类原因": error or "标注字段objects为空",
            })
            continue

        raw_label_set = set()
        for obj in objects:
            if isinstance(obj, dict) and obj.get("name"):
                raw_label_set.update(_split_object_labels(obj.get("name")))
        raw_label_combo = "，".join(sorted(raw_label_set)) if raw_label_set else ""
        row_expand_count = 0
        row_reason_set = set()
        any_classified = False

        for obj in objects:
            if not isinstance(obj, dict): continue
            raw_name = obj.get("name")
            labels = _split_object_labels(raw_name)
            if not labels:
                row_copy = row.copy()
                row_copy["无法分类原因"] = "标注框缺少name字段"
                unclassified_rows.append(row_copy)
                continue

            for label in labels:
                if label not in label_to_category:
                    row_copy = row.copy()
                    row_copy["无法分类原因"] = f"标签{label}未在规则中定义"
                    row_copy["无法分类标签"] = label
                    unclassified_rows.append(row_copy)
                    row_reason_set.add(f"标签{label}未在规则中定义")
                    continue

                category = label_to_category[label]
                new_row = row.copy()
                obj_copy = copy.deepcopy(obj)
                obj_copy["name"] = label
                new_data = {k: v for k, v in data.items() if k != "objects"}
                new_data["objects"] = [obj_copy]
                new_json = json.dumps(new_data, ensure_ascii=False)
                for col in json_columns:
                    if col in df.columns:
                        new_row[col] = new_json
                new_row["分类标签"] = label
                new_row["分类类别"] = category
                new_row["原始标签组合"] = raw_label_combo
                category_rows.setdefault(category, []).append(new_row)
                any_classified = True
                row_expand_count += 1

        if not any_classified:
            row_copy = row.copy()
            row_copy["无法分类原因"] = "；".join(sorted(row_reason_set)) if row_reason_set else "标签无法匹配规则"
            unclassified_rows.append(row_copy)

        status = "部分可分类" if row_reason_set else "是"
        if not any_classified:
            status = "否"
        
        split_counts_rows.append({
            "source": row.get("source"),
            "原始标签组合": raw_label_combo,
            "拆分条数": row_expand_count,
            "是否可分类": status,
            "无法分类原因": "；".join(sorted(row_reason_set)),
        })

    category_files = []
    category_counts = {}
    for category, rows in category_rows.items():
        if not rows: continue
        category_counts[category] = len(rows)
        cat_df = pd.DataFrame(rows)
        cat_df = cat_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        n_total = len(cat_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_df = cat_df.iloc[:n_train]
        val_df = cat_df.iloc[n_train:n_train + n_val]
        test_df = cat_df.iloc[n_train + n_val:]
        safe_name = _safe_filename(category)
        out_path = output_dir / f"{safe_name}.xlsx"
        with pd.ExcelWriter(out_path) as writer:
            train_df.to_excel(writer, sheet_name="train", index=False)
            val_df.to_excel(writer, sheet_name="val", index=False)
            test_df.to_excel(writer, sheet_name="test", index=False)
        category_files.append(out_path)

    unclassified_path = output_dir / "unclassified.xlsx"
    pd.DataFrame(unclassified_rows).to_excel(unclassified_path, index=False)
    split_counts_path = output_dir / "split_counts.xlsx"
    pd.DataFrame(split_counts_rows).to_excel(split_counts_path, index=False)

    return {
        "output_dir": output_dir,
        "category_files": category_files,
        "unclassified": unclassified_path,
        "split_counts": split_counts_path,
        "summary": {
            "categories": len(category_rows),
            "classified": sum(category_counts.values()),
            "unclassified": len(unclassified_rows),
            "category_counts": category_counts,
        },
    }

def summarize_unclassified(
        unclassified_excel_path: str,
        output_dir: str,
        json_columns: Optional[list] = None,
):
    if not os.path.exists(unclassified_excel_path):
        raise FileNotFoundError(f"无法分类文件不存在：{unclassified_excel_path}")

    df = pd.read_excel(unclassified_excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if json_columns is None:
        json_columns = []
        if "新_结果字段-目标检测标签配置" in df.columns:
            json_columns.append("新_结果字段-目标检测标签配置")
        if "结果字段-目标检测标签配置" in df.columns:
            json_columns.append("结果字段-目标检测标签配置")

    reason_col = "无法分类原因"
    if reason_col not in df.columns:
        df[reason_col] = "未知原因"

    reason_counts = df[reason_col].fillna("未知原因").value_counts().reset_index()
    reason_counts.columns = ["原因", "数量"]

    label_counter = {}
    reason_label_counter = {}
    reason_label_pattern = re.compile(r"^标签(.+?)(未在规则中定义)$")

    for _, row in df.iterrows():
        reason = row.get(reason_col, "未知原因")
        labels = []
        if "无法分类标签" in df.columns:
            labels = _split_object_labels(row.get("无法分类标签"))

        if not labels:
            match = reason_label_pattern.match(str(reason))
            if match:
                labels = [match.group(1)]
            else:
                label_counter["无标签"] = label_counter.get("无标签", 0) + 1
                reason_label_counter[("无标签", reason)] = reason_label_counter.get(("无标签", reason), 0) + 1
                continue

        for label in labels:
            label_counter[label] = label_counter.get(label, 0) + 1
            reason_label_counter[(label, reason)] = reason_label_counter.get((label, reason), 0) + 1

    label_summary = pd.DataFrame([{"标签": k, "数量": v} for k, v in label_counter.items()]).sort_values("数量", ascending=False)
    reason_label_summary = pd.DataFrame([{"标签": k[0], "原因": k[1], "数量": v} for k, v in reason_label_counter.items()]).sort_values("数量", ascending=False)

    out_path = output_dir / "unclassified_summary.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        reason_counts.to_excel(writer, sheet_name="reason_summary", index=False)
        label_summary.to_excel(writer, sheet_name="label_summary", index=False)
        reason_label_summary.to_excel(writer, sheet_name="reason_label", index=False)

    return out_path

def generate_yolo_datasets_from_excels(
        category_excels: list,
        output_dir: str,
        image_cache_dir: Optional[str] = None,
        source_col: str = "source",
        label_col: str = "分类标签",
        json_col_primary: str = "新_结果字段-目标检测标签配置",
        json_col_fallback: str = "结果字段-目标检测标签配置",
        width_col: str = "width",
        height_col: str = "height",
        download_images: bool = True,
        random_seed: int = 42,
        class_order: Optional[list] = None,
        resume: bool = True,
        progress_callback=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(image_cache_dir) if image_cache_dir else (output_dir / "image_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    dataset_name_map = {}
    skipped = []
    dataset_stats = {}
    total_rows = 0
    processed_rows = 0
    downloaded_images = 0

    used_dir_names = set()
    
    # Pre-calculate total rows
    for excel_path in category_excels:
        if not excel_path or not Path(excel_path).exists(): continue
        xls = pd.ExcelFile(excel_path)
        for split in ["train", "val", "test"]:
            if split in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=split)
                total_rows += len(df)

    for idx_excel, excel_path in enumerate(category_excels):
        if not excel_path or not Path(excel_path).exists(): continue
        excel_path = Path(excel_path)
        category_name = excel_path.stem
        base_dir_name = _safe_dataset_dir_name(category_name, f"category_{idx_excel:03d}")
        dir_name = base_dir_name
        suffix = 1
        while dir_name in used_dir_names:
            dir_name = f"{base_dir_name}_{suffix}"
            suffix += 1
        used_dir_names.add(dir_name)
        dataset_dir = output_dir / dir_name
        dataset_name_map[dataset_dir.name] = category_name
        
        images_root = dataset_dir / "images"
        labels_root = dataset_dir / "labels"
        for split in ["train", "val", "test"]:
            (images_root / split).mkdir(parents=True, exist_ok=True)
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        xls = pd.ExcelFile(excel_path)
        split_sheets = [s for s in ["train", "val", "test"] if s in xls.sheet_names]
        
        all_labels = []
        split_dfs = {}
        for split in split_sheets:
            df_split = pd.read_excel(excel_path, sheet_name=split)
            split_dfs[split] = df_split
            if label_col in df_split.columns:
                all_labels.extend([str(v) for v in df_split[label_col].dropna()])
        
        classes = sorted(list(dict.fromkeys(all_labels)))
        if class_order:
            ordered = [c for c in class_order if c in classes]
            remaining = [c for c in classes if c not in ordered]
            classes = ordered + remaining
        class_to_id = {name: i for i, name in enumerate(classes)}

        dataset_stats[category_name] = {"train": 0, "val": 0, "test": 0}
        
        for split in split_sheets:
            df_split = split_dfs[split]
            df_split = df_split.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            for idx, row in df_split.iterrows():
                current_info = (processed_rows, total_rows, downloaded_images, category_name, split, f"idx_{idx}", "", excel_path.name, idx)
                if progress_callback and processed_rows % 50 == 0:
                    progress_callback(*current_info)

                source = row.get(source_col)
                if not source:
                    skipped.append({"category": category_name, "reason": "缺少source", "split": split})
                    processed_rows += 1
                    continue
                
                label_value = str(row.get(label_col, ""))
                if not label_value or label_value not in class_to_id:
                    skipped.append({"category": category_name, "reason": "缺少或无效分类标签", "split": split})
                    processed_rows += 1
                    continue

                image_stem = _safe_image_stem(str(source), idx)
                label_path = labels_root / split / f"{image_stem}.txt"
                
                if resume and label_path.exists() and label_path.stat().st_size > 0:
                     dataset_stats[category_name][split] += 1
                     processed_rows += 1
                     continue

                json_str = row.get(json_col_primary) or row.get(json_col_fallback)
                boxes = _extract_boxes_with_labels(json_str)
                filtered_boxes = [b for b in boxes if b[0] == label_value]
                
                if not filtered_boxes:
                    skipped.append({"category": category_name, "reason": "无匹配标签框", "split": split})
                    processed_rows += 1
                    continue
                
                image_path = None
                if download_images:
                    image_path = _ensure_image_cached(str(source), cache_dir)
                elif Path(str(source)).exists():
                    image_path = Path(str(source))
                
                width = row.get(width_col)
                height = row.get(height_col)
                if (not width or not height) and image_path:
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                    except: pass
                
                if not width or not height:
                    skipped.append({"category": category_name, "reason": "缺少图像尺寸", "split": split})
                    processed_rows += 1
                    continue

                out_image = images_root / split / f"{image_stem}{image_path.suffix if image_path else '.jpg'}"
                if image_path:
                    if not out_image.exists():
                        try:
                            out_image.write_bytes(Path(image_path).read_bytes())
                            downloaded_images += 1
                        except:
                            skipped.append({"category": category_name, "reason": "图片写入失败", "split": split})
                            processed_rows += 1
                            continue
                else:
                    skipped.append({"category": category_name, "reason": "图片下载失败", "split": split})
                    processed_rows += 1
                    continue

                label_lines = []
                for _, x1, y1, x2, y2 in filtered_boxes:
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    bw = max(x2 - x1, 0.0)
                    bh = max(y2 - y1, 0.0)
                    if bw <= 0 or bh <= 0: continue
                    label_lines.append(f"{class_to_id[label_value]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {bw / width:.6f} {bh / height:.6f}")

                if label_lines:
                    label_path.write_text("\n".join(label_lines), encoding="utf-8")
                    dataset_stats[category_name][split] += 1
                else:
                    skipped.append({"category": category_name, "reason": "标注框无效", "split": split})
                
                processed_rows += 1

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(yaml.dump({
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": classes
        }, sort_keys=False, allow_unicode=True), encoding="utf-8")
        datasets.append(dataset_dir)

    skipped_path = output_dir / "yolo_skipped.xlsx"
    pd.DataFrame(skipped if skipped else [{"category": "无", "reason": "无", "split": "无"}]).to_excel(skipped_path, index=False)
    
    if progress_callback:
        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)

    return {
        "datasets": datasets,
        "skipped": skipped_path,
        "stats": dataset_stats,
        "total": total_rows,
        "processed": processed_rows,
        "downloaded": downloaded_images,
        "dataset_name_map": dataset_name_map,
    }

def summarize_yolo_label_counts(dataset_dirs):
    stats = {}
    flat_rows = []
    for dataset_dir in dataset_dirs or []:
        if not dataset_dir: continue
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists(): continue
        
        names = []
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                names = data.get("names") or []
            except Exception: pass
            
        dataset_key = dataset_path.name
        split_stats = {}
        total_images_all = 0
        total_img_counts = {}
        total_box_counts = {}
        
        for split in ["train", "val", "test"]:
            label_dir = dataset_path / "labels" / split
            img_counts = {}
            box_counts = {}
            total_images = 0
            if label_dir.exists():
                for txt_path in label_dir.glob("*.txt"):
                    total_images += 1
                    try:
                        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    except: continue
                    labels_in_image = set()
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        try:
                            class_id = int(float(parts[0]))
                            label_name = names[class_id] if class_id < len(names) else str(class_id)
                            labels_in_image.add(label_name)
                            box_counts[label_name] = box_counts.get(label_name, 0) + 1
                        except: continue
                    for label in labels_in_image:
                        img_counts[label] = img_counts.get(label, 0) + 1
            
            split_stats[split] = {"total_images": total_images, "label_counts": img_counts, "box_counts": box_counts}
            total_images_all += total_images
            for label, count in img_counts.items():
                total_img_counts[label] = total_img_counts.get(label, 0) + count
            for label, count in box_counts.items():
                total_box_counts[label] = total_box_counts.get(label, 0) + count

            for label in set(img_counts) | set(box_counts):
                flat_rows.append({
                    "数据集": dataset_key, "split": split, "标签": label,
                    "图片数量": img_counts.get(label, 0),
                    "标注框数量": box_counts.get(label, 0),
                    "占比%": f"{(img_counts.get(label, 0) / total_images * 100):.1f}%" if total_images else "0.0%",
                    "split总图片数": total_images
                })
        
        split_stats["all"] = {"total_images": total_images_all, "label_counts": total_img_counts, "box_counts": total_box_counts}
        stats[dataset_key] = split_stats
        
        for label in set(total_img_counts) | set(total_box_counts):
            flat_rows.append({
                "数据集": dataset_key, "split": "all", "标签": label,
                "图片数量": total_img_counts.get(label, 0),
                "标注框数量": total_box_counts.get(label, 0),
                "占比%": f"{(total_img_counts.get(label, 0) / total_images_all * 100):.1f}%" if total_images_all else "0.0%",
                "split总图片数": total_images_all
            })

    return stats, pd.DataFrame(flat_rows)
