import io
import importlib.util
import json
import os
import hashlib
import math
import time
import queue
import random
import re
import threading
import shutil
import zipfile
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# === Merged utils/display.py ===
import json
import math

import pandas as pd
import streamlit as st


def safe_dataframe(df: pd.DataFrame, **kwargs):
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
    try:
        st.dataframe(safe_df, **kwargs)
    except Exception:
        fallback = safe_df.astype(str, errors="ignore")
        st.dataframe(fallback, **kwargs)
        st.caption("已将数据转换为文本以便展示。")
# === Merged process_step.py ===
import json
import copy
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
import re


def merge_all_csv_in_folder(
        folder_path,
        output_file="merged_csv.csv",
        encoding="utf-8-sig",
        chunk_size: int = 100000,
        progress_callback=None,
):
    """
    合并指定文件夹下的所有CSV文件

    :param folder_path: 包含CSV文件的文件夹路径
    :param output_file: 合并后的输出文件名（默认"merged_csv.csv"）
    :param encoding: 文件编码（默认"utf-8-sig"，兼容中文和BOM头）
    :param chunk_size: 分块读取行数（默认100000）
    :param progress_callback: 进度回调（file_idx, total_files, filename, total_rows, file_rows, chunk_idx）
    :return: 合并后的总行数
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在：{folder_path}")

    # 获取文件夹中所有CSV文件的路径
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
            # 分块读取CSV文件，避免一次性占满内存
            file_rows = 0
            with open(csv_file, "r", encoding=encoding, errors="ignore") as f:
                chunk_iter = pd.read_csv(
                    f,
                    parse_dates=False,  # 避免自动解析日期导致错误
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


# # ---------------------- 使用示例 ----------------------
# if __name__ == "__main__":
#     # 替换为你的CSV文件夹路径
#     csv_folder = "标注结果 2"
#     # 合并CSV（输出文件默认在当前目录，可自定义路径）
#     merge_all_csv_in_folder(
#         folder_path=csv_folder,
#         output_file="merged_result.csv"
#     )


def deduplicate_csv_by_source(
        csv_path: str,
        output_file: Optional[str] = "deduplicate_result.csv",
        encoding: str = "utf-8-sig",
        keep: str = "first",
        verbose: bool = True
) -> pd.DataFrame:
    """
    读取CSV文件，并根据source列去重

    :param csv_path: CSV文件的路径（绝对路径或相对路径）
    :param output_file: 去重后的输出文件路径（默认None：不保存文件，仅返回DataFrame）
    :param encoding: 文件编码（默认"utf-8-sig"，兼容中文和BOM头；中文文件可尝试"gbk"）
    :param keep: 去重保留策略："first"（保留第一条重复数据，默认）、"last"（保留最后一条）
    :param verbose: 是否输出详细日志（默认True）
    :return: 去重后的DataFrame
    """
    # 1. 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

    # 2. 检查文件后缀是否为CSV
    if not csv_path.endswith(".csv"):
        raise ValueError(f"文件不是CSV格式：{csv_path}（请传入.csv后缀的文件）")

    # 3. 读取CSV文件
    try:
        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            parse_dates=False  # 避免自动解析日期导致错误
        )
        if verbose:
            print(f"成功读取CSV文件：{os.path.basename(csv_path)}")
            print(f"读取后原始数据行数：{len(df)}")
    except Exception as e:
        raise Exception(f"读取CSV文件失败：{str(e)}") from e

    # 4. 检查source列是否存在
    if "source" not in df.columns:
        raise KeyError(f"CSV文件中未找到'source'列，请检查列名是否正确（当前列名：{list(df.columns)}）")

    # 5. 执行去重
    original_count = len(df)
    deduplicated_df = df.drop_duplicates(
        subset=["source"],  # 按source列去重
        keep=keep,  # 保留策略
        ignore_index=True  # 重置索引（避免索引断裂）
    )
    duplicate_count = original_count - len(deduplicated_df)

    # 6. 输出去重日志
    if verbose:
        print(f"去重策略：按'source'列保留{keep}条数据")
        print(f"去除重复数据行数：{duplicate_count}")
        print(f"去重后剩余数据行数：{len(deduplicated_df)}")

    # 7. 保存去重后的文件（如果指定output_file）
    if output_file is not None:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            deduplicated_df.to_csv(output_file, index=False, encoding=encoding)
            if verbose:
                print(f"去重后的文件已保存至：{os.path.abspath(output_file)}")
        except Exception as e:
            raise Exception(f"保存去重文件失败：{str(e)}") from e

    return deduplicated_df

# # # 读取CSV并按source列去重（保留第一条）
# df = deduplicate_csv_by_source(
#     csv_path="merged_result.csv",  # 你的CSV文件路径
#     keep="first"
# )
# # 后续可对df进行其他操作（如筛选、统计）
# print(df["source"].value_counts())  # 验证去重结果（每个source仅出现1次）



def remove_duplicates_between_csv(
        main_csv: str,
        ref_csv: str,
        output_csv: str = "filtered_main.csv",
        compare_col: str = "source",
        encoding: str = "utf-8-sig",
        verbose: bool = True
) -> pd.DataFrame:
    """
    对比两个CSV文件，剔除「主CSV中在参考CSV中出现过的记录」，生成无重复的新CSV

    核心逻辑：以 `compare_col`（默认source列）为基准，保留主CSV中该列值不在参考CSV中的所有记录

    :param main_csv: 主CSV路径（需要剔除重复数据的CSV）
    :param ref_csv: 参考CSV路径（用于判断重复的基准CSV）
    :param output_csv: 输出无重复数据的新CSV路径（默认"filtered_main.csv"）
    :param compare_col: 用于对比去重的列名（默认"source"，需两个CSV中都存在该列）
    :param encoding: 文件编码（默认"utf-8-sig"，兼容中文和BOM头；中文可尝试"gbk"）
    :param verbose: 是否输出详细日志（默认True）
    :return: 剔除重复后的数据（DataFrame）
    """
    # 1. 检查文件是否存在
    for csv_path in [main_csv, ref_csv]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在：{csv_path}")
        if not csv_path.endswith(".csv"):
            raise ValueError(f"文件不是CSV格式：{csv_path}（请传入.csv后缀文件）")

    # 2. 读取两个CSV文件
    try:
        # 读取主CSV
        df_main = pd.read_csv(
            main_csv,
            encoding=encoding,
            parse_dates=False
        )
        # 读取参考CSV
        df_ref = pd.read_csv(
            ref_csv,
            encoding=encoding,
            parse_dates=False
        )
        if verbose:
            print(f"成功读取文件：")
            print(f"- 主CSV（{os.path.basename(main_csv)}）：{len(df_main)} 行")
            print(f"- 参考CSV（{os.path.basename(ref_csv)}）：{len(df_ref)} 行")
    except Exception as e:
        raise Exception(f"读取CSV失败：{str(e)}") from e

    # 3. 检查对比列是否存在于两个CSV中
    for df, df_name in [(df_main, "主CSV"), (df_ref, "参考CSV")]:
        if compare_col not in df.columns:
            raise KeyError(
                f"{df_name}中未找到对比列「{compare_col}」\n"
                f"{df_name}现有列名：{list(df.columns)}"
            )

    # 4. 提取参考CSV的对比列唯一值（用于快速判断重复）
    ref_unique_vals = set(df_ref[compare_col].dropna())  # 去重+排除NaN值
    if verbose:
        print(f"参考CSV中「{compare_col}」列共有 {len(ref_unique_vals)} 个唯一值")

    # 5. 剔除主CSV中与参考CSV重复的记录
    # 保留：主CSV中对比列值不在参考CSV中的行
    df_filtered = df_main[~df_main[compare_col].isin(ref_unique_vals)].reset_index(drop=True)
    duplicate_count = len(df_main) - len(df_filtered)

    # 6. 输出去重统计
    if verbose:
        print(f"\n去重结果：")
        print(f"- 主CSV原始行数：{len(df_main)}")
        print(f"- 剔除重复行数：{duplicate_count}")
        print(f"- 剩余有效行数：{len(df_filtered)}")

    # 7. 保存剔除重复后的新CSV
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        df_filtered.to_csv(output_csv, index=False, encoding=encoding)
        if verbose:
            print(f"\n无重复数据已保存至：{os.path.abspath(output_csv)}")
    except Exception as e:
        raise Exception(f"保存输出CSV失败：{str(e)}") from e

    return df_filtered


# # # 主CSV：需要剔除重复的文件
# # 参考CSV：用于判断重复的基准文件
# # 输出：剔除重复后的新文件（默认filtered_main.csv）
# remove_duplicates_between_csv(
#     main_csv="deduplicate_result.csv",
#     ref_csv="reference.csv"
# )


def overwrite_reference_with_result(
        result_csv: str = "deduplicate_result.csv",
        reference_csv: str = "reference.csv",
        encoding: str = "utf-8-sig",
        backup_original: bool = True,
        verbose: bool = True
) -> None:
    """
    清空reference.csv原有数据，将deduplicate_result.csv的数据完整写入reference.csv

    核心逻辑：
    1. 校验输入文件（result_csv必须存在且为CSV，reference_csv不存在则自动创建）
    2. 可选：备份reference.csv原有数据（避免误操作丢失）
    3. 读取result_csv数据
    4. 覆盖写入reference.csv（清空原有内容）

    :param result_csv: 数据源CSV路径（默认"deduplicate_result.csv"）
    :param reference_csv: 目标CSV路径（默认"reference.csv"）
    :param encoding: 文件编码（默认"utf-8-sig"，兼容中文和BOM头）
    :param backup_original: 是否备份reference.csv原有数据（默认True，备份文件名为reference_backup_时间戳.csv）
    :param verbose: 是否输出详细日志（默认True）
    """
    # 1. 校验数据源文件（result_csv必须存在且为CSV）
    if not os.path.exists(result_csv):
        raise FileNotFoundError(f"数据源文件不存在：{result_csv}")
    if not result_csv.endswith(".csv"):
        raise ValueError(f"数据源文件不是CSV格式：{result_csv}（请传入.csv后缀文件）")

    # 2. 备份reference.csv原有数据（如果需要且文件已存在）
    if backup_original and os.path.exists(reference_csv):
        from datetime import datetime
        # 生成带时间戳的备份文件名（避免覆盖历史备份）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_csv = f"{os.path.splitext(reference_csv)[0]}_backup_{timestamp}.csv"
        try:
            # 复制原有数据到备份文件
            df_original = pd.read_csv(reference_csv, encoding=encoding, parse_dates=False)
            df_original.to_csv(backup_csv, index=False, encoding=encoding)
            if verbose:
                print(f"✅ 已备份reference.csv原有数据至：{backup_csv}（{len(df_original)}行）")
        except Exception as e:
            raise Exception(f"备份reference.csv失败：{str(e)}") from e

    # 3. 读取deduplicate_result.csv数据
    try:
        df_result = pd.read_csv(
            result_csv,
            encoding=encoding,
            parse_dates=False
        )
        if verbose:
            print(f"✅ 成功读取数据源文件：{os.path.basename(result_csv)}（{len(df_result)}行数据）")
    except Exception as e:
        raise Exception(f"读取数据源文件{result_csv}失败：{str(e)}") from e

    # 4. 覆盖写入reference.csv（清空原有内容，写入新数据）
    try:
        # 确保输出目录存在（如果reference.csv在子目录中）
        output_dir = os.path.dirname(reference_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 覆盖写入（index=False不保留索引列）
        df_result.to_csv(reference_csv, index=False, encoding=encoding, mode="w")
        if verbose:
            print(f"✅ 已成功覆盖reference.csv：")
            print(f"   - 原数据已{'备份' if backup_original else '未备份'}")
            print(f"   - 新写入数据行数：{len(df_result)}")
            print(f"   - 目标文件路径：{os.path.abspath(reference_csv)}")
    except Exception as e:
        raise Exception(f"写入reference.csv失败：{str(e)}") from e


# # # ---------------------- 执行程序 ----------------------
# if __name__ == "__main__":
#     try:
#         # 调用函数执行操作（使用默认参数，可根据需要修改路径）
#         overwrite_reference_with_result(
#             result_csv="deduplicate_result.csv",  # 数据源文件（可改为绝对路径）
#             reference_csv="reference.csv",  # 目标文件（可改为绝对路径）
#             encoding="utf-8-sig",  # 编码（中文文件可改为"gbk"）
#             backup_original=True  # 建议保留备份，避免数据丢失
#         )
#         print("\n🎉 操作完成！")
#     except Exception as e:
#         print(f"\n❌ 操作失败：{str(e)}")



def process_csv_replace_ptlist(
        input_csv_path,
        output_csv_path="processed_replaced_ptlist.csv",
        excluded_output_file="processed_replaced_ptlist_excluded.csv"
):
    try:
        # 1. 读取原始CSV文件
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

        # 2. 检查必要列是否存在
        required_cols = ["source", "结果字段-目标检测标签配置", "是否废弃"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误：原始CSV缺少必要列：{', '.join(missing_cols)}")
            return

        # 3. 筛选条件："是否废弃"为"否" 且 "结果字段-目标检测标签配置"非空
        cond_not_discarded = df["是否废弃"] == "否"
        cond_has_json = (df["结果字段-目标检测标签配置"].notna()) & (df["结果字段-目标检测标签配置"] != "")
        filtered_df = df[cond_not_discarded & cond_has_json].copy()

        # 3.1 记录未筛选数据及原因
        excluded_df = df[~(cond_not_discarded & cond_has_json)].copy()
        if not excluded_df.empty:
            reasons = []
            for _, row in excluded_df.iterrows():
                row_reasons = []
                if row.get("是否废弃") != "否":
                    row_reasons.append("是否废弃不为否")
                val = row.get("结果字段-目标检测标签配置")
                if pd.isna(val) or val == "":
                    row_reasons.append("标注字段为空")
                elif not isinstance(val, str):
                    row_reasons.append("标注字段非字符串")
                if not row_reasons:
                    row_reasons.append("未满足筛选条件")
                reasons.append("；".join(row_reasons))
            excluded_df["未筛选原因"] = reasons

        # 4. 处理source列：替换开头的"oss"为"http"（仅匹配开头）
        filtered_df["source"] = filtered_df["source"].str.replace("^oss", "http", n=1, regex=True)

        # 5. 定义单个ptList的最小包围盒计算函数（返回两个点坐标）
        def get_bbox_points(ptlist):
            """输入单个ptList，返回最小包围盒的两个点：左上(x1,y1)、右下(x2,y2)"""
            if not isinstance(ptlist, list) or len(ptlist) == 0:
                return [{"x": None, "y": None}, {"x": None, "y": None}]

            # 提取有效坐标点
            valid_points = [p for p in ptlist if isinstance(p, dict) and "x" in p and "y" in p]
            if not valid_points:
                return [{"x": None, "y": None}, {"x": None, "y": None}]

            # 计算包围盒坐标
            min_x = min(p["x"] for p in valid_points)
            max_x = max(p["x"] for p in valid_points)
            min_y = min(p["y"] for p in valid_points)
            max_y = max(p["y"] for p in valid_points)

            # 返回两个点（左上、右下）
            return [{"x": min_x, "y": min_y}, {"x": max_x, "y": max_y}]

        # 6. 解析原始JSON，替换每个object的ptList为包围盒两点，保留其他字段
        def parse_and_replace_ptlist(json_str):
            """解析原始JSON，替换ptList为包围盒两点，生成新JSON字符串"""
            try:
                if pd.isna(json_str) or not isinstance(json_str, str):
                    return None

                # 解析原始JSON数据
                data = json.loads(json_str)
                objects = data.get("objects", [])

                # 遍历每个object，替换ptList
                updated_objects = []
                for obj in objects:
                    if isinstance(obj, dict):
                        # 深拷贝原始object，避免修改原数据
                        updated_obj = obj.copy()
                        # 获取原始ptList
                        original_ptlist = obj.get("polygon", {}).get("ptList", [])
                        # 计算包围盒两点，替换原始ptList
                        updated_ptlist = get_bbox_points(original_ptlist)
                        # 更新polygon中的ptList
                        if "polygon" not in updated_obj:
                            updated_obj["polygon"] = {}
                        updated_obj["polygon"]["ptList"] = updated_ptlist
                        updated_objects.append(updated_obj)

                # 替换objects为更新后的数据
                data["objects"] = updated_objects
                # 转换为JSON字符串返回
                return json.dumps(data, ensure_ascii=False)

            except json.JSONDecodeError:
                print(f"警告：JSON解析失败（截取前50字符）：{json_str[:50]}...")
                return None

        # 7. 生成新的"结果字段-目标检测标签配置"列（替换ptList后）
        filtered_df["新_结果字段-目标检测标签配置"] = filtered_df["结果字段-目标检测标签配置"].apply(
            parse_and_replace_ptlist)

        # 8. 提取width、height字段（可选，保持与之前逻辑一致）
        def extract_width_height(json_str):
            """从JSON中提取width和height"""
            try:
                if pd.isna(json_str) or not isinstance(json_str, str):
                    return {"width": None, "height": None}
                data = json.loads(json_str)
                return {"width": data.get("width"), "height": data.get("height")}
            except:
                return {"width": None, "height": None}

        # 应用提取函数
        wh_data = filtered_df["结果字段-目标检测标签配置"].apply(extract_width_height)
        filtered_df["width"] = [item["width"] for item in wh_data]
        filtered_df["height"] = [item["height"] for item in wh_data]

        # 9. 定义最终保留的列（原始列+新JSON列+提取字段）
        result_cols = [
            "source",  # 处理后的URL列
            "结果字段-目标检测标签配置",  # 原始JSON列（保留）
            "新_结果字段-目标检测标签配置",  # 替换ptList后的新JSON列
            "width", "height"  # 提取的图片尺寸字段
        ]

        # 10. 保存新CSV文件
        filtered_df[result_cols].to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print(f"处理完成！新文件已保存到：{output_csv_path}")
        print(f"筛选后有效行数：{len(filtered_df)}")
        if excluded_output_file is not None:
            excluded_df.to_csv(excluded_output_file, index=False, encoding="utf-8-sig")
            print(f"未筛选数据已保存到：{excluded_output_file}（{len(excluded_df)}行）")
        print("关键变化：每个object的ptList已替换为最小包围盒的两个点（左上、右下）")
        print("保留字段：原始JSON列、处理后的URL列、width、height，新JSON列结构与原始一致")

    except FileNotFoundError:
        print(f"错误：未找到输入文件：{input_csv_path}")
    except Exception as e:
        print(f"处理异常：{e}")

    return {
        "filtered_rows": len(filtered_df) if "filtered_df" in locals() else 0,
        "excluded_rows": len(excluded_df) if "excluded_df" in locals() else 0,
        "excluded_output": excluded_output_file,
    }





# ####step1
# # # ---------------------- 请修改以下参数 ----------------------
# input_csv_path = "deduplicate_result.csv"  # 替换为你的原始CSV文件路径
# # ------------------------------------------------------------
#
# # 调用函数（默认输出文件名为 processed_result.csv，可自定义）
# process_csv_replace_ptlist(input_csv_path)







def filter_by_box_count_and_iou(input_csv_path,
                                high_iou_csv="high_iou_0.98.csv",
                                other_csv="other_data.csv",
                                min_boxes: int = 2,
                                iou_threshold: float = 0.98):
    """
    按「标注框数量」和「两框IoU阈值」筛选数据，并拆分为高IoU与其他数据。
    """
    # 1. 计算两个矩形框的交并比（IoU）
    def calculate_iou(box1, box2):
        """
        计算两个矩形框的IoU
        :param box1: 第一个框坐标 (x1, y1, x2, y2)
        :param box2: 第二个框坐标 (x1, y1, x2, y2)
        :return: IoU值（0~1）
        """
        # 计算重叠区域坐标
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # 重叠面积（无重叠则为0）
        intersection = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if intersection == 0:
            return 0.0

        # 两个框的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 计算IoU
        union = area1 + area2 - intersection
        return intersection / union if union != 0 else 0.0

    # 2. 从JSON中提取标注框（适配新结果字段的两点格式）
    def extract_boxes(json_str):
        """
        从"新_结果字段-目标检测标签配置"提取所有标注框坐标
        :return: 框列表 [(x1,y1,x2,y2), ...]，仅保留有效两点框
        """
        boxes = []
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return boxes
            data = json.loads(json_str)
            objects = data.get("objects", [])

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                # 提取两点坐标（新结果字段应为两点包围盒）
                ptlist = obj.get("polygon", {}).get("ptList", [])
                if len(ptlist) != 2:
                    continue  # 只处理标准两点框
                # 解析x、y坐标（确保有效）
                p1, p2 = ptlist
                if not (isinstance(p1, dict) and isinstance(p2, dict)
                        and "x" in p1 and "y" in p1
                        and "x" in p2 and "y" in p2):
                    continue
                # 规范化坐标（确保x1 < x2，y1 < y2）
                x1 = min(p1["x"], p2["x"])
                y1 = min(p1["y"], p2["y"])
                x2 = max(p1["x"], p2["x"])
                y2 = max(p1["y"], p2["y"])
                boxes.append((x1, y1, x2, y2))
        except json.JSONDecodeError:
            print(f"警告：JSON解析失败（前50字符）：{str(json_str)[:50]}...")
        except Exception as e:
            print(f"提取框失败：{e}")
        return boxes

    # 3. 检查是否满足条件：标注框≥min_boxes 且 存在两框IoU≥iou_threshold
    def meet_conditions(boxes):
        """
        :param boxes: 标注框列表
        :return: True（满足条件）/ False（不满足）
        """
        # 条件1：标注框数量≥min_boxes
        if len(boxes) < min_boxes:
            return False
        # 条件2：存在任意两框IoU≥iou_threshold
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = calculate_iou(boxes[i], boxes[j])
                if iou >= iou_threshold:
                    return True
        return False

    # 4. 读取CSV并筛选数据
    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
        print(f"成功读取CSV，共 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_csv_path}")
        return
    except Exception as e:
        print(f"读取失败：{e}")
        return

    # 检查必要列
    required_col = "新_结果字段-目标检测标签配置"
    if required_col not in df.columns:
        print(f"错误：缺少必要列 {required_col}")
        return

    # 5. 分离符合条件和不符合条件的数据
    high_iou_data = []  # 符合条件：框≥min_boxes 且 IoU≥iou_threshold
    other_data = []  # 其他数据

    for idx, row in df.iterrows():
        json_str = row[required_col]
        boxes = extract_boxes(json_str)
        if meet_conditions(boxes):
            high_iou_data.append(row)
        else:
            other_data.append(row)

    # 6. 保存结果（关键修改：为空时保留表头）
    # 保存符合条件的数据（high_iou_csv）
    high_iou_df = pd.DataFrame(high_iou_data, columns=df.columns)  # 强制指定表头
    high_iou_df.to_csv(high_iou_csv, index=False, encoding="utf-8-sig")

    # 保存其他数据（other_csv）：为空时仍保留表头
    other_df = pd.DataFrame(other_data, columns=df.columns)  # 强制指定表头
    other_df.to_csv(other_csv, index=False, encoding="utf-8-sig")

    # 输出统计
    print(f"\n筛选完成！")
    print(f"符合条件（框≥{min_boxes} 且 IoU≥{iou_threshold}）：{len(high_iou_data)} 行 → {high_iou_csv}")
    print(f"其他数据：{len(other_data)} 行 → {other_csv}")
    print(f"注：若某文件行数为0，已自动保留原CSV表头")


# # ---------------------- 配置参数 ----------------------
# input_csv_path = "processed_replaced_ptlist.csv"  # 你的输入CSV路径
# # ------------------------------------------------------
#
# # 执行筛选
# filter_by_box_count_and_iou(input_csv_path)


def download_and_draw_annotations(
        input_csv_path,
        output_dir: Optional[str] = None,
        download_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
        max_images: Optional[int] = None,
        timeout: int = 15
):
    # 1. 定义文件夹路径，自动创建不存在的文件夹
    base_dir = Path(output_dir) if output_dir else Path(os.getcwd())
    download_dir = Path(download_dir) if download_dir else (base_dir / "downloaded_images")
    result_dir = Path(result_dir) if result_dir else (base_dir / "annotated_images")
    download_dir.mkdir(parents=True, exist_ok=True)  # 自动创建文件夹，已存在则忽略
    result_dir.mkdir(parents=True, exist_ok=True)

    # 2. 读取CSV文件
    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
        print(f"成功读取CSV文件，共 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"错误：未找到CSV文件 {input_csv_path}")
        return
    except Exception as e:
        print(f"读取CSV失败：{e}")
        return

    # 3. 检查必要列是否存在
    required_cols = ["source", "结果字段-目标检测标签配置", "新_结果字段-目标检测标签配置"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：CSV缺少必要列：{', '.join(missing_cols)}")
        return

    # 4. 定义字体（用于绘制类别名称，避免中文乱码）
    def get_font():
        """尝试加载系统字体，失败则使用默认字体"""
        try:
            # Windows系统
            return ImageFont.truetype("simhei.ttf", 48)
        except:
            try:
                # Mac系统
                return ImageFont.truetype("Arial Unicode.ttf", 48)
            except:
                #  fallback：使用默认字体
                return ImageFont.load_default()

    font = get_font()

    # 5. 定义绘制标注框的函数
    def draw_annotation_boxes(image, json_str, color, draw):
        """
        在图片上绘制标注框和类别名称
        :param image: PIL.Image对象
        :param json_str: 标注JSON字符串
        :param color: 标注框颜色（RGB元组）
        :param draw: ImageDraw对象
        """
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return
            data = json.loads(json_str)
            objects = data.get("objects", [])

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                # 提取类别名称
                name = obj.get("name", "未知类别")
                # 提取ptList（坐标点）
                ptlist = obj.get("polygon", {}).get("ptList", [])
                if not ptlist or len(ptlist) < 2:
                    continue

                # 解析坐标点（适配原始多边形/新包围盒两点）
                points = []
                for p in ptlist:
                    if (isinstance(p, dict) and "x" in p and "y" in p
                            and p["x"] is not None and p["y"] is not None):
                        points.append((p["x"], p["y"]))

                if len(points) < 2:
                    continue

                # 绘制标注框：两点则画矩形，多点则画多边形
                if len(points) == 2:
                    # 两点模式（左上、右下）：画矩形
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    # 在矩形左上角绘制类别名称（背景半透明）
                    text_bbox = draw.textbbox((x1, y1 - 20), name, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255, 180))  # 白色半透明背景
                    draw.text((x1, y1 - 20), name, font=font, fill=color)
                else:
                    # 多点模式（原始多边形）：画多边形
                    draw.polygon(points, outline=color, width=2)
                    # 在多边形左上角附近绘制类别名称
                    min_x = min(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    text_bbox = draw.textbbox((min_x, min_y - 20), name, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255, 180))
                    draw.text((min_x, min_y - 20), name, font=font, fill=color)

        except json.JSONDecodeError:
            print(f"警告：标注JSON解析失败")
        except Exception as e:
            print(f"绘制标注失败：{e}")

    # 6. 遍历每行数据，下载图片并绘制标注
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        processed_count = success_count + fail_count
        if max_images is not None and processed_count >= max_images:
            print(f"\n已达到最大处理数量：{max_images}")
            break
        source_url = row["source"]
        original_anno = row["结果字段-目标检测标签配置"]
        new_anno = row["新_结果字段-目标检测标签配置"]

        # 生成图片文件名（从URL提取或用索引命名）
        img_filename = source_url.split("/")[-1] if "/" in source_url else f"image_{idx}.jpg"
        download_path = download_dir / img_filename
        result_path = result_dir / img_filename

        # 跳过已下载的图片（避免重复下载）
        if not os.path.exists(download_path):
            print(f"\n正在下载图片：{source_url}")
            try:
                # 下载图片
                response = requests.get(source_url, stream=True, timeout=timeout)
                response.raise_for_status()  # 捕获HTTP错误
                with open(download_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"下载成功：{img_filename}")
            except requests.exceptions.RequestException as e:
                print(f"下载失败：{e}")
                fail_count += 1
                continue
        else:
            print(f"\n图片已存在，跳过下载：{img_filename}")

        # 打开图片并绘制标注
        try:
            with Image.open(download_path) as img:
                draw = ImageDraw.Draw(img)
                # 绘制原始标注框（红色：RGB(255,0,0)）
                draw_annotation_boxes(img, original_anno, (255, 0, 0), draw)
                # 绘制新标注框（绿色：RGB(0,255,0)）
                draw_annotation_boxes(img, new_anno, (0, 255, 0), draw)
                # 保存标注后的图片
                img.save(result_path)
            print(f"标注完成：{img_filename}")
            success_count += 1
        except Exception as e:
            print(f"图片处理失败：{e}")
            fail_count += 1

    # 7. 输出统计结果
    print(f"\n处理完成！")
    print(f"成功处理：{success_count} 张图片")
    print(f"失败：{fail_count} 张图片")
    print(f"下载图片路径：{download_dir}")
    print(f"标注图片路径：{result_dir}")


def _safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\\s+", "_", name)
    return name or "unknown"


def _to_pinyin_slug(text: str) -> Optional[str]:
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return None
    parts = []
    for ch in str(text):
        if re.match(r"[\u4e00-\u9fff]", ch):
            py = lazy_pinyin(ch)
            if py:
                parts.append(py[0])
        else:
            parts.append(ch)
    slug = "".join(parts).strip()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug.lower() if slug else None


def _ascii_safe_filename(name: str, fallback: str) -> str:
    base = _safe_filename(name)
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    return ascii_name or fallback


def _safe_dataset_dir_name(name: str, fallback: str) -> str:
    text = str(name).strip()
    if re.search(r"[\u4e00-\u9fff]", text):
        pinyin_slug = _to_pinyin_slug(text)
        if pinyin_slug:
            return pinyin_slug
    return _ascii_safe_filename(text, fallback)


def _split_label_cell(value: str) -> list:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    # 每个单元格即一个标签，不做拆分
    return [text]


def _parse_objects(json_str: str):
    if pd.isna(json_str) or not isinstance(json_str, str):
        return None, "标注字段为空或非字符串"
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, "标注字段JSON解析失败"
    objects = data.get("objects")
    if objects is None:
        return None, "标注字段缺少objects"
    if not isinstance(objects, list) or len(objects) == 0:
        return [], "标注字段objects为空"
    return objects, None


def _parse_data_objects(json_str: str):
    if pd.isna(json_str) or not isinstance(json_str, str):
        return None, None, "标注字段为空或非字符串"
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, None, "标注字段JSON解析失败"
    objects = data.get("objects")
    if objects is None:
        return data, None, "标注字段缺少objects"
    if not isinstance(objects, list) or len(objects) == 0:
        return data, [], "标注字段objects为空"
    return data, objects, None


def _split_object_labels(label_str: str) -> list:
    if label_str is None:
        return []
    text = str(label_str).strip()
    if not text or text.lower() == "nan":
        return []
    # 仅支持中文逗号/英文逗号分隔的多标签
    parts = re.split(r"[，,]+", text)
    return [p.strip() for p in parts if p.strip()]


def _replace_label_tokens(raw_name: str, mapping: dict) -> tuple:
    if raw_name is None:
        return raw_name, 0, 0
    text = str(raw_name).strip()
    if not text:
        return raw_name, 0, 0
    labels = _split_object_labels(text)
    if not labels:
        return raw_name, 0, 0
    replaced = 0
    new_labels = []
    for lbl in labels:
        new_lbl = mapping.get(lbl, lbl)
        if new_lbl != lbl:
            replaced += 1
        new_labels.append(new_lbl)
    if "，" in text:
        joiner = "，"
    elif "," in text:
        joiner = ","
    else:
        joiner = "，"
    return joiner.join(new_labels), replaced, len(labels)


def _filter_json_by_label(json_str: str, label: str):
    if pd.isna(json_str) or not isinstance(json_str, str):
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    objects = data.get("objects", [])
    filtered = [obj for obj in objects if isinstance(obj, dict) and obj.get("name") == label]
    if not filtered:
        return None
    data["objects"] = filtered
    return json.dumps(data, ensure_ascii=False)


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
    """
    使用新旧标签对照表替换标注中的标签名称。
    支持在标注字段的 objects[].name 中替换，输出新的CSV。
    """
    df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

    mapping_df = pd.read_excel(mapping_excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(mapping_excel_path)
    if mapping_df.empty:
        raise ValueError("标签对照表为空")

    if not old_col or not new_col:
        cols = list(mapping_df.columns)
        if len(cols) < 2:
            raise ValueError("标签对照表至少需要两列（旧标签、新标签）")
        old_col = old_col or cols[0]
        new_col = new_col or cols[1]

    label_map = {}
    for _, row in mapping_df.iterrows():
        old_label = str(row.get(old_col, "")).strip()
        new_label = str(row.get(new_col, "")).strip()
        if not old_label or old_label.lower() == "nan":
            continue
        if not new_label or new_label.lower() == "nan":
            continue
        label_map[old_label] = new_label
    if not label_map:
        raise ValueError("标签对照表未包含有效映射")

    if json_columns is None:
        json_columns = []
        if "新_结果字段-目标检测标签配置" in df.columns:
            json_columns.append("新_结果字段-目标检测标签配置")
        if "结果字段-目标检测标签配置" in df.columns:
            json_columns.append("结果字段-目标检测标签配置")
    if not json_columns:
        raise KeyError("输入CSV中未找到标注字段列")

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
        diff_df = pd.DataFrame(diff_rows)
        diff_df.to_excel(diff_path, index=False)

    unmatched_path = None
    if unmatched_excel_path:
        unmatched_path = Path(unmatched_excel_path)
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        if unmatched_counter:
            unmatched_df = pd.DataFrame(
                [{"标签": k, "数量": v} for k, v in unmatched_counter.items()]
            ).sort_values("数量", ascending=False)
        else:
            unmatched_df = pd.DataFrame(columns=["标签", "数量"])
        unmatched_df.to_excel(unmatched_path, index=False)

    sample_diff = []
    for item in diff_rows[: sample_size or 0]:
        sample_diff.append(item)

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
        "sample_diff": sample_diff,
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
    """
    根据分类规则将数据拆分为多个类别Excel，并生成 train/val/test。
    - 支持宽表（类别为列）和两列映射（标签-类别）。
    - 多标签数据会生成多条记录，每条只保留一个标签的标注框。
    - 无法分类的数据单独输出并附原因。
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"输入CSV不存在：{input_csv_path}")
    if not os.path.exists(rules_excel_path):
        raise FileNotFoundError(f"规则Excel不存在：{rules_excel_path}")

    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("训练/验证/测试比例之和必须大于0")

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
    if not json_columns:
        raise KeyError("输入CSV中未找到标注字段列")

    rules_df = pd.read_excel(rules_excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(rules_excel_path)

    label_to_category = {}
    label_conflicts = {}

    if rule_mode == "wide":
        for col in rules_df.columns:
            category = str(col).strip()
            if not category:
                continue
            for cell in rules_df[col].dropna():
                labels = _split_label_cell(cell)
                for label in labels:
                    if label in label_to_category and label_to_category[label] != category:
                        label_conflicts.setdefault(label, set()).update([label_to_category[label], category])
                    else:
                        label_to_category[label] = category
    elif rule_mode == "two_column":
        if not label_col or not category_col:
            raise ValueError("两列映射模式需要提供 label_col 和 category_col")
        for _, row in rules_df.iterrows():
            label = str(row.get(label_col, "")).strip()
            category = str(row.get(category_col, "")).strip()
            if not label or not category or label.lower() == "nan" or category.lower() == "nan":
                continue
            if label in label_to_category and label_to_category[label] != category:
                label_conflicts.setdefault(label, set()).update([label_to_category[label], category])
            else:
                label_to_category[label] = category
    else:
        raise ValueError("rule_mode 仅支持 wide 或 two_column")

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
        if error:
            row_copy = row.copy()
            row_copy["无法分类原因"] = error
            unclassified_rows.append(row_copy)
            split_counts_rows.append({
                "source": row.get("source"),
                "原始标签组合": "",
                "拆分条数": 0,
                "是否可分类": "否",
                "无法分类原因": error,
            })
            continue

        if not objects:
            row_copy = row.copy()
            row_copy["无法分类原因"] = "标注字段objects为空"
            unclassified_rows.append(row_copy)
            split_counts_rows.append({
                "source": row.get("source"),
                "原始标签组合": "",
                "拆分条数": 0,
                "是否可分类": "否",
                "无法分类原因": "标注字段objects为空",
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
            if not isinstance(obj, dict):
                continue
            raw_name = obj.get("name")
            labels = _split_object_labels(raw_name)
            if not labels:
                row_copy = row.copy()
                row_copy["无法分类原因"] = "标注框缺少name字段"
                row_copy["无法分类标签"] = ""
                unclassified_rows.append(row_copy)
                row_reason_set.add("标注框缺少name字段")
                continue

            for label in labels:
                # if label in label_conflicts:
                #     row_copy = row.copy()
                #     row_copy["无法分类原因"] = f"标签{label}在规则中映射多个类别"
                #     row_copy["无法分类标签"] = label
                #     unclassified_rows.append(row_copy)
                #     row_reason_set.add(f"标签{label}在规则中映射多个类别")
                #     continue
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
            reason_msg = "；".join(sorted(row_reason_set)) if row_reason_set else "标签无法匹配规则"
            row_copy["无法分类原因"] = reason_msg
            unclassified_rows.append(row_copy)

        reason_msg = "；".join(sorted(row_reason_set)) if row_reason_set else ""
        if any_classified:
            status = "部分可分类" if reason_msg else "是"
        else:
            status = "否"
            if not reason_msg:
                reason_msg = "标签无法匹配规则"
        split_counts_rows.append({
            "source": row.get("source"),
            "原始标签组合": raw_label_combo,
            "拆分条数": row_expand_count,
            "是否可分类": status,
            "无法分类原因": reason_msg,
        })

    category_files = []
    category_counts = {}
    for category, rows in category_rows.items():
        if not rows:
            continue
        category_counts[category] = len(rows)
        cat_df = pd.DataFrame(rows)
        cat_df = cat_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        n_total = len(cat_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
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
    if unclassified_rows:
        unclassified_df = pd.DataFrame(unclassified_rows)
        unclassified_df.to_excel(unclassified_path, index=False)
    else:
        pd.DataFrame(columns=list(df.columns) + ["无法分类原因"]).to_excel(unclassified_path, index=False)

    split_counts_path = output_dir / "split_counts.xlsx"
    if split_counts_rows:
        pd.DataFrame(split_counts_rows).to_excel(split_counts_path, index=False)
    else:
        pd.DataFrame(columns=["source", "原始标签组合", "拆分条数", "是否可分类", "无法分类原因"]).to_excel(split_counts_path, index=False)

    classified_total = sum(category_counts.values())
    return {
        "output_dir": output_dir,
        "category_files": category_files,
        "unclassified": unclassified_path,
        "split_counts": split_counts_path,
        "summary": {
            "categories": len(category_rows),
            "classified": classified_total,
            "unclassified": len(unclassified_rows),
            "category_counts": category_counts,
        },
    }


def summarize_unclassified(
        unclassified_excel_path: str,
        output_dir: str,
        json_columns: Optional[list] = None,
):
    """
    汇总无法分类数据的原因与标签统计，输出到单个Excel文件。
    """
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

    # reason_label_pattern = re.compile(r"^标签(.+?)(未在规则中定义|在规则中映射多个类别)$")
    reason_label_pattern = re.compile(r"^标签(.+?)(未在规则中定义)$")

    for _, row in df.iterrows():
        reason = row.get(reason_col, "未知原因")
        json_str = None
        for col in json_columns:
            if col in row and isinstance(row[col], str) and row[col]:
                json_str = row[col]
                break

        # 优先使用“无法分类标签”列
        labels = []
        if "无法分类标签" in df.columns:
            raw_unclassified = row.get("无法分类标签")
            labels = _split_object_labels(raw_unclassified)

        if not labels:
            objects, error = _parse_objects(json_str)
            if objects is None or error:
                # 尝试从原因中提取标签
                match = reason_label_pattern.match(str(reason))
                if match:
                    labels = [match.group(1)]
                else:
                    label_counter["无标签"] = label_counter.get("无标签", 0) + 1
                    reason_label_counter[("无标签", reason)] = reason_label_counter.get(("无标签", reason), 0) + 1
                    continue
            else:
                for obj in objects:
                    if isinstance(obj, dict) and obj.get("name"):
                        labels.extend(_split_object_labels(obj.get("name")))
                labels = list(dict.fromkeys(labels))
                if not labels:
                    label_counter["无标签"] = label_counter.get("无标签", 0) + 1
                    reason_label_counter[("无标签", reason)] = reason_label_counter.get(("无标签", reason), 0) + 1
                    continue

        for label in labels:
            label_counter[label] = label_counter.get(label, 0) + 1
            reason_label_counter[(label, reason)] = reason_label_counter.get((label, reason), 0) + 1

    label_summary = pd.DataFrame(
        [{"标签": k, "数量": v} for k, v in label_counter.items()]
    ).sort_values("数量", ascending=False)

    reason_label_summary = pd.DataFrame(
        [{"标签": k[0], "原因": k[1], "数量": v} for k, v in reason_label_counter.items()]
    ).sort_values("数量", ascending=False)

    out_path = output_dir / "unclassified_summary.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        reason_counts.to_excel(writer, sheet_name="reason_summary", index=False)
        label_summary.to_excel(writer, sheet_name="label_summary", index=False)
        reason_label_summary.to_excel(writer, sheet_name="reason_label", index=False)

    return out_path


def _extract_boxes_with_labels(json_str: str):
    if pd.isna(json_str) or not isinstance(json_str, str):
        return []
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []
    objects = data.get("objects", [])
    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = obj.get("name")
        ptlist = obj.get("polygon", {}).get("ptList", [])
        points = []
        for p in ptlist:
            if isinstance(p, dict) and "x" in p and "y" in p and p["x"] is not None and p["y"] is not None:
                points.append((p["x"], p["y"]))
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append((label, x1, y1, x2, y2))
    return boxes


def _safe_image_stem(source: str, idx: int) -> str:
    if not source:
        return f"image_{idx}"
    name = str(source).split("/")[-1].split("?")[0]
    name = os.path.splitext(name)[0]
    name = _safe_filename(name)
    return f"{idx:06d}_{name}"


def _ensure_image_cached(source: str, cache_dir: Path, timeout: int = 15):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = str(source).split("/")[-1].split("?")[0]
    if not filename:
        filename = f"image_{abs(hash(source))}.jpg"
    filename = _safe_filename(filename)
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path
    if str(source).startswith("http"):
        try:
            resp = requests.get(source, stream=True, timeout=timeout)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return cache_path
        except Exception:
            return None
    else:
        src_path = Path(source)
        if src_path.exists():
            try:
                cache_path.write_bytes(src_path.read_bytes())
                return cache_path
            except Exception:
                return None
    return None


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
    """
    根据分类后的Excel生成YOLO格式数据集。
    每个类别Excel生成一个数据集目录，包含 images/labels/train|val|test 和 data.yaml。
    """
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

    current_category = None
    current_split = None
    current_file = None
    current_label = None
    current_excel = None
    current_row_idx = None

    used_dir_names = set()
    for idx_excel, excel_path in enumerate(category_excels):
        if not excel_path or not Path(excel_path).exists():
            continue
        excel_path = Path(excel_path)
        current_excel = excel_path.name
        category_name = excel_path.stem
        current_category = category_name
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
        if not split_sheets:
            continue

        all_labels = []
        split_dfs = {}
        for split in split_sheets:
            df_split = pd.read_excel(excel_path, sheet_name=split)
            split_dfs[split] = df_split
            if label_col in df_split.columns:
                all_labels.extend([str(v) for v in df_split[label_col].dropna()])
            total_rows += len(df_split)

        classes = sorted(list(dict.fromkeys(all_labels)))
        if class_order:
            ordered = [c for c in class_order if c in classes]
            remaining = [c for c in classes if c not in ordered]
            classes = ordered + remaining
        class_to_id = {name: i for i, name in enumerate(classes)}

        dataset_stats[category_name] = {"train": 0, "val": 0, "test": 0}
        if progress_callback:
            progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
        for split in split_sheets:
            current_split = split
            df_split = split_dfs[split]
            df_split = df_split.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            for idx, row in df_split.iterrows():
                current_row_idx = idx
                current_file = _safe_image_stem(str(row.get(source_col)), idx)
                source = row.get(source_col)
                if not source:
                    skipped.append({"category": category_name, "reason": "缺少source", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                label_value = row.get(label_col)
                if not label_value:
                    skipped.append({"category": category_name, "reason": "缺少分类标签", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue
                label_value = str(label_value)
                current_label = label_value
                if label_value not in class_to_id:
                    skipped.append({"category": category_name, "reason": "标签未在类别列表中", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                image_stem = _safe_image_stem(str(source), idx)
                current_file = image_stem
                label_path = labels_root / split / f"{image_stem}.txt"
                if resume and label_path.exists() and label_path.stat().st_size > 0:
                    existing_images = list((images_root / split).glob(f"{image_stem}.*"))
                    if existing_images:
                        dataset_stats[category_name][split] += 1
                        processed_rows += 1
                        if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                            progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                        continue

                json_str = None
                if json_col_primary in row and isinstance(row[json_col_primary], str) and row[json_col_primary]:
                    json_str = row[json_col_primary]
                elif json_col_fallback in row and isinstance(row[json_col_fallback], str) and row[json_col_fallback]:
                    json_str = row[json_col_fallback]

                boxes = _extract_boxes_with_labels(json_str)
                if not boxes:
                    skipped.append({"category": category_name, "reason": "标注框为空", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                filtered_boxes = [b for b in boxes if b[0] == label_value]
                if not filtered_boxes:
                    skipped.append({"category": category_name, "reason": "无匹配标签框", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                width = row.get(width_col)
                height = row.get(height_col)

                if (not width or not height) and json_str:
                    try:
                        data = json.loads(json_str)
                        width = width or data.get("width")
                        height = height or data.get("height")
                    except Exception:
                        pass

                image_path = None
                if download_images:
                    image_path = _ensure_image_cached(str(source), cache_dir)
                else:
                    if Path(str(source)).exists():
                        image_path = Path(str(source))

                if image_path and (not width or not height):
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                    except Exception:
                        pass

                if not width or not height:
                    skipped.append({"category": category_name, "reason": "缺少图像尺寸", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                image_suffix = ".jpg"
                if image_path:
                    image_suffix = image_path.suffix if image_path.suffix else ".jpg"
                image_name = f"{image_stem}{image_suffix}"
                out_image = images_root / split / image_name

                if image_path:
                    try:
                        wrote_new = False
                        if not out_image.exists():
                            out_image.write_bytes(Path(image_path).read_bytes())
                            wrote_new = True
                        if wrote_new:
                            downloaded_images += 1
                    except Exception:
                        skipped.append({"category": category_name, "reason": "图片写入失败", "split": split})
                        processed_rows += 1
                        if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                            progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                        continue
                else:
                    skipped.append({"category": category_name, "reason": "图片下载失败", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                label_lines = []
                for _, x1, y1, x2, y2 in filtered_boxes:
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    bw = max(x2 - x1, 0.0)
                    bh = max(y2 - y1, 0.0)
                    if bw <= 0 or bh <= 0:
                        continue
                    cx = (x1 + x2) / 2 / width
                    cy = (y1 + y2) / 2 / height
                    bw_n = bw / width
                    bh_n = bh / height
                    class_id = class_to_id[label_value]
                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")

                if not label_lines:
                    skipped.append({"category": category_name, "reason": "标注框无效", "split": split})
                    processed_rows += 1
                    if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                        progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)
                    continue

                label_path = labels_root / split / f"{image_stem}.txt"
                label_path.write_text("\n".join(label_lines), encoding="utf-8")
                dataset_stats[category_name][split] += 1
                processed_rows += 1
                if progress_callback and (processed_rows % 50 == 0 or processed_rows == total_rows):
                    progress_callback(processed_rows, total_rows, downloaded_images, current_category, current_split, current_file, current_label, current_excel, current_row_idx)

        data_yaml = dataset_dir / "data.yaml"
        names_json = json.dumps(classes, ensure_ascii=False)
        data_yaml.write_text(
            "\n".join([
                f"path: {dataset_dir}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(classes)}",
                f"names: {names_json}",
            ]),
            encoding="utf-8",
        )

        datasets.append(dataset_dir)

    skipped_path = output_dir / "yolo_skipped.xlsx"
    if skipped:
        pd.DataFrame(skipped).to_excel(skipped_path, index=False)
    else:
        pd.DataFrame(columns=["category", "reason", "split"]).to_excel(skipped_path, index=False)

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
# # # ---------------------- 需手动修改的参数 ----------------------
# input_csv_path = "other_data.csv"  # 替换为你生成的CSV文件路径
# # --------------------------------------------------------------
#
# # 调用函数
# download_and_draw_annotations(input_csv_path)




# import pandas as pd
# import requests
# import json
# import os
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# import yaml  # 需要额外安装pyyaml库
#
#
# def generate_yolo_dataset(input_csv_path,
#                           yolo_root="yolo_dataset",
#                           val_split=0.2,
#                           class_mapping=None):
#     # 1. 初始化类别映射和ID
#     if class_mapping is None:
#         class_mapping = {}
#         class_id = 0  # 自动分配ID时初始化
#     else:
#         # 手动指定映射时，从最大ID+1开始
#         class_id = max(class_mapping.values(), default=-1) + 1
#
#     # 创建文件夹结构
#     train_img_dir = os.path.join(yolo_root, "train", "images")
#     train_label_dir = os.path.join(yolo_root, "train", "labels")
#     val_img_dir = os.path.join(yolo_root, "val", "images")
#     val_label_dir = os.path.join(yolo_root, "val", "labels")
#     for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
#         Path(dir_path).mkdir(parents=True, exist_ok=True)
#
#     # 2. 图片下载函数
#     def download_image(url, save_path):
#         if os.path.exists(save_path):
#             return True
#         try:
#             response = requests.get(url, stream=True, timeout=15)
#             response.raise_for_status()
#             with open(save_path, "wb") as f:
#                 f.write(response.content)
#             return True
#         except Exception as e:
#             print(f"图片下载失败 {url}：{e}")
#             return False
#
#     # 3. 标注转换函数
#     def json_to_yolo_annotation(json_str, img_width, img_height):
#         nonlocal class_mapping, class_id
#         yolo_lines = []
#         try:
#             if pd.isna(json_str) or not isinstance(json_str, str):
#                 return ""
#             data = json.loads(json_str)
#             objects = data.get("objects", [])
#             img_width = img_width or data.get("width", 1)
#             img_height = img_height or data.get("height", 1)
#
#             for obj in objects:
#                 if not isinstance(obj, dict):
#                     continue
#                 # 处理类别ID
#                 obj_name = obj.get("name", "unknown")
#                 if obj_name not in class_mapping:
#                     class_mapping[obj_name] = class_id
#                     class_id += 1
#                 cid = class_mapping[obj_name]
#
#                 # 处理坐标
#                 ptlist = obj.get("polygon", {}).get("ptList", [])
#                 if len(ptlist) != 2:
#                     continue
#                 p1, p2 = ptlist
#                 x1, y1 = min(p1["x"], p2["x"]), min(p1["y"], p2["y"])
#                 x2, y2 = max(p1["x"], p2["x"]), max(p1["y"], p2["y"])
#
#                 # 转换为YOLO格式（归一化）
#                 x_center = (x1 + x2) / 2 / img_width
#                 y_center = (y1 + y2) / 2 / img_height
#                 width = (x2 - x1) / img_width
#                 height = (y2 - y1) / img_height
#
#                 # 限制范围在0~1之间
#                 x_center = max(0.001, min(0.999, x_center))
#                 y_center = max(0.001, min(0.999, y_center))
#                 width = max(0.001, min(0.999, width))
#                 height = max(0.001, min(0.999, height))
#
#                 yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
#
#         except json.JSONDecodeError:
#             print(f"JSON解析失败（前50字符）：{str(json_str)[:50]}...")
#         except Exception as e:
#             print(f"标注转换失败：{e}")
#
#         return "\n".join(yolo_lines)
#
#     # 4. 读取并处理CSV
#     try:
#         df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
#         print(f"成功读取CSV，共 {len(df)} 行数据")
#     except Exception as e:
#         print(f"CSV处理失败：{e}")
#         return
#
#     # 5. 划分训练集和验证集
#     train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
#     print(f"训练集：{len(train_df)} 条 | 验证集：{len(val_df)} 条")
#
#     # 6. 处理数据集（图片+标注）
#     def process_split(df, img_dir, label_dir, split_name):
#         success = 0
#         fail = 0
#         for idx, row in df.iterrows():
#             img_url = row["source"]
#             json_anno = row["新_结果字段-目标检测标签配置"]
#             img_width = row.get("width")
#             img_height = row.get("height")
#
#             # 生成文件名
#             img_filename = os.path.splitext(os.path.basename(img_url))[0] + ".jpg"
#             label_filename = os.path.splitext(img_filename)[0] + ".txt"
#
#             # 下载图片
#             img_path = os.path.join(img_dir, img_filename)
#             if not download_image(img_url, img_path):
#                 fail += 1
#                 continue
#
#             # 生成标注
#             yolo_anno = json_to_yolo_annotation(json_anno, img_width, img_height)
#             with open(os.path.join(label_dir, label_filename), "w", encoding="utf-8") as f:
#                 f.write(yolo_anno)
#
#             success += 1
#         print(f"{split_name}集：成功 {success} 条 | 失败 {fail} 条")
#
#     process_split(train_df, train_img_dir, train_label_dir, "训练")
#     process_split(val_df, val_img_dir, val_label_dir, "验证")
#
#     # 7. 生成类别映射文件（yolo_classes.txt）
#     class_file = os.path.join(yolo_root, "yolo_classes.txt")
#     with open(class_file, "w", encoding="utf-8") as f:
#         for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
#             f.write(f"{cls_id} {cls_name}\n")
#
#     # 8. 生成YOLO训练所需的yaml文件（关键新增功能）
#     yaml_data = {
#         "path": os.path.abspath(yolo_root),  # 数据集根目录绝对路径
#         "train": "train/images",  # 训练集图片相对路径
#         "val": "val/images",  # 验证集图片相对路径
#         "nc": len(class_mapping),  # 类别数量
#         "names": [cls_name for cls_name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]  # 类别名称列表
#     }
#
#     yaml_path = os.path.join(yolo_root, "dataset.yaml")
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)  # 保留中文且不排序键
#
#     # 9. 输出最终结果
#     print(f"\n数据集生成完成！根目录：{os.path.abspath(yolo_root)}")
#     print(f"类别数：{len(class_mapping)} | 类别文件：{class_file}")
#     print(f"YOLO配置文件：{yaml_path}（可直接用于训练）")
#
#
# # ---------------------- 配置参数 ----------------------
# input_csv_path = "other_data.csv"  # 替换为你的CSV路径
# val_split = 0.2  # 验证集比例
# class_mapping = {
#     # 示例：手动指定类别ID（可选）
#     # "国槐|生物圈|虫害|国槐尺蠖|严重": 0,
#     # "国槐|生物圈|虫害|国槐尺蠖|明显": 1
# }
# # ------------------------------------------------------
#
# # 安装依赖（若未安装）
# # pip install pandas requests pillow scikit-learn pyyaml
#
# # 生成数据集
# generate_yolo_dataset(input_csv_path, val_split=val_split, class_mapping=class_mapping)

import csv
import json
from collections import defaultdict
from typing import Optional, List


def process_detection_data(
        input_csv: str,
        output_matched: str,
        output_unmatched: str,
        target_classes: Optional[List[str]] = None,
        min_total_count: int = 0  # 改为“全量数据中类别总次数>阈值”
) -> None:
    """
    处理目标检测数据：统计类别数量 + 按条件筛选数据
    筛选逻辑：先判断目标类别在全量数据中的总次数是否>min_total_count，
              再筛选出包含这些“达标类别”的所有行

    Args:
        input_csv: 输入CSV文件路径（other_data.csv）
        output_matched: 符合条件的数据保存路径
        output_unmatched: 不符合条件的数据保存路径
        target_classes: 目标检测类别列表（如["国槐|本体|主干", "国槐|本体|树冠"]），None表示不限制类别
        min_total_count: 全量数据中类别的最小总次数（需大于该值）
    """
    # 校验参数
    if min_total_count < 0:
        raise ValueError("最小总次数阈值不能为负数")
    if target_classes is None:
        target_classes = []

    # 第一步：统计所有类别的全量总次数
    class_counter = defaultdict(int)
    total_rows = 0
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if '新_结果字段-目标检测标签配置' not in reader.fieldnames:
            raise KeyError("CSV文件中缺少'新_结果字段-目标检测标签配置'列")

        for row in reader:
            total_rows += 1
            try:
                config_str = row['新_结果字段-目标检测标签配置'].strip() or '{}'
                config = json.loads(config_str)
                objects = config.get('objects', [])
                # 统计每个类别的全量总次数（不限制单行重复）
                for obj in objects:
                    class_name = obj.get('name', '未知类别').strip()
                    if class_name:
                        class_counter[class_name] += 1
            except json.JSONDecodeError:
                print(f"警告：第{total_rows}行JSON格式错误，跳过统计")
            except Exception as e:
                print(f"警告：第{total_rows}行统计失败（{str(e)}），跳过")

    # 打印统计结果
    print("\n===== 检测目标类别统计 =====")
    print(f"总数据行数：{total_rows}")
    valid_classes = {k: v for k, v in class_counter.items() if k.strip()}
    print(f"检测到的有效类别总数：{len(valid_classes)}")
    for class_name, count in sorted(valid_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}次")
    print("===========================\n")

    # 第二步：确定“达标类别”（目标类别中总次数>min_total_count的类别）
    qualified_classes = []
    if target_classes:
        # 筛选出目标类别中总次数达标的类别
        for cls in target_classes:
            cls = cls.strip()
            total_count = class_counter.get(cls, 0)
            if total_count > min_total_count:
                qualified_classes.append(cls)
                print(f"类别「{cls}」达标（总次数={total_count} > {min_total_count}）")
            else:
                print(f"类别「{cls}」未达标（总次数={total_count} ≤ {min_total_count}）")
    else:
        # 不限制类别时，所有总次数>min_total_count的类别都算达标
        qualified_classes = [cls for cls, count in class_counter.items() if count > min_total_count]
        print(f"不限制类别，达标类别数量：{len(qualified_classes)}（总次数>{min_total_count}）")

    if not qualified_classes:
        print(f"\n无达标类别，所有数据归入未匹配文件")
        # 直接复制所有数据到未匹配文件
        with open(input_csv, 'r', encoding='utf-8') as infile, \
                open(output_unmatched, 'w', encoding='utf-8', newline='') as unmatchfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(unmatchfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(reader)
        return

    # 第三步：筛选出包含“达标类别”的所有行
    with open(input_csv, 'r', encoding='utf-8') as infile, \
            open(output_matched, 'w', encoding='utf-8', newline='') as matchfile, \
            open(output_unmatched, 'w', encoding='utf-8', newline='') as unmatchfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        match_writer = csv.DictWriter(matchfile, fieldnames=fieldnames)
        unmatch_writer = csv.DictWriter(unmatchfile, fieldnames=fieldnames)
        match_writer.writeheader()
        unmatch_writer.writeheader()

        matched_count = 0

        for row in reader:
            try:
                config_str = row['新_结果字段-目标检测标签配置'].strip() or '{}'
                config = json.loads(config_str)
                objects = config.get('objects', [])
                # 检查当前行是否包含任何一个达标类别
                row_has_qualified = False
                for obj in objects:
                    class_name = obj.get('name', '').strip()
                    if class_name in qualified_classes:
                        row_has_qualified = True
                        break  # 找到一个就够，无需继续检查

                if row_has_qualified:
                    match_writer.writerow(row)
                    matched_count += 1
                else:
                    unmatch_writer.writerow(row)

            except json.JSONDecodeError:
                unmatch_writer.writerow(row)
                print(f"警告：第{reader.line_num}行JSON格式错误，归入未匹配数据")
            except Exception as e:
                unmatch_writer.writerow(row)
                print(f"警告：第{reader.line_num}行筛选失败（{str(e)}），归入未匹配数据")

    # 输出筛选结果统计
    print(f"\n筛选完成：")
    print(f"达标类别：{qualified_classes}")
    print(f"符合条件的数据（包含达标类别）：{matched_count}条（已保存至{output_matched}）")
    print(f"不符合条件的数据：{total_rows - matched_count}条（已保存至{output_unmatched}）")


# 使用示例（你的需求：类别=['国槐|本体|主干', '国槐|本体|树冠']，总次数>500）
# if __name__ == "__main__":
#     process_detection_data(
#         input_csv="other_data.csv",
#         output_matched="matched_data.csv",
#         output_unmatched="unmatched_data.csv",
#         target_classes=[],#"国槐|本体|主干", "国槐|本体|树冠"
#         min_total_count=0  # 全量总次数>500
#     )

#
# import pandas as pd
# import requests
# import json
# import os
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# import yaml
# from tqdm import tqdm  # 用于显示进度条
#
# def generate_yolo_dataset(input_csv_path,
#                           yolo_root="yolo_dataset",
#                           val_split=0.2,
#                           class_mapping=None):
#     # 1. 初始化类别映射和ID
#     if class_mapping is None:
#         class_mapping = {}
#         class_id = 0  # 自动分配ID时初始化
#     else:
#         # 手动指定映射时，从最大ID+1开始
#         class_id = max(class_mapping.values(), default=-1) + 1
#
#     # 创建文件夹结构
#     train_img_dir = os.path.join(yolo_root, "train", "images")
#     train_label_dir = os.path.join(yolo_root, "train", "labels")
#     val_img_dir = os.path.join(yolo_root, "val", "images")
#     val_label_dir = os.path.join(yolo_root, "val", "labels")
#     for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
#         Path(dir_path).mkdir(parents=True, exist_ok=True)
#
#     # 2. 图片下载函数
#     def download_image(url, save_path):
#         if os.path.exists(save_path):
#             return True
#         try:
#             response = requests.get(url, stream=True, timeout=15)
#             response.raise_for_status()
#             with open(save_path, "wb") as f:
#                 f.write(response.content)
#             return True
#         except Exception as e:
#             print(f"\n图片下载失败 {url}：{e}")  # 换行避免打断进度条
#             return False
#
#     # 3. 标注转换函数
#     def json_to_yolo_annotation(json_str, img_width, img_height):
#         nonlocal class_mapping, class_id
#         yolo_lines = []
#         try:
#             if pd.isna(json_str) or not isinstance(json_str, str):
#                 return ""
#             data = json.loads(json_str)
#             objects = data.get("objects", [])
#             img_width = img_width or data.get("width", 1)
#             img_height = img_height or data.get("height", 1)
#
#             for obj in objects:
#                 if not isinstance(obj, dict):
#                     continue
#                 # 处理类别ID
#                 obj_name = obj.get("name", "unknown")
#                 if obj_name not in class_mapping:
#                     class_mapping[obj_name] = class_id
#                     class_id += 1
#                 cid = class_mapping[obj_name]
#
#                 # 处理坐标
#                 ptlist = obj.get("polygon", {}).get("ptList", [])
#                 if len(ptlist) != 2:
#                     continue
#                 p1, p2 = ptlist
#                 x1, y1 = min(p1["x"], p2["x"]), min(p1["y"], p2["y"])
#                 x2, y2 = max(p1["x"], p2["x"]), max(p1["y"], p2["y"])
#
#                 # 转换为YOLO格式（归一化）
#                 x_center = (x1 + x2) / 2 / img_width
#                 y_center = (y1 + y2) / 2 / img_height
#                 width = (x2 - x1) / img_width
#                 height = (y2 - y1) / img_height
#
#                 # 限制范围在0~1之间
#                 x_center = max(0.001, min(0.999, x_center))
#                 y_center = max(0.001, min(0.999, y_center))
#                 width = max(0.001, min(0.999, width))
#                 height = max(0.001, min(0.999, height))
#
#                 yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
#
#         except json.JSONDecodeError:
#             print(f"\nJSON解析失败（前50字符）：{str(json_str)[:50]}...")
#         except Exception as e:
#             print(f"\n标注转换失败：{e}")
#
#         return "\n".join(yolo_lines)
#
#     # 4. 读取并处理CSV
#     try:
#         df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
#         print(f"成功读取CSV，共 {len(df)} 行数据")
#     except Exception as e:
#         print(f"CSV处理失败：{e}")
#         return
#
#     # 5. 划分训练集和验证集
#     train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
#     print(f"训练集：{len(train_df)} 条 | 验证集：{len(val_df)} 条")
#
#     # 6. 处理数据集（图片+标注），添加进度条
#     def process_split(df, img_dir, label_dir, split_name):
#         success = 0
#         fail = 0
#         # 使用tqdm创建进度条，desc显示当前处理阶段
#         for idx, row in tqdm(enumerate(df.iterrows()),
#                              total=len(df),
#                              desc=f"处理{split_name}集",
#                              unit="条"):
#             _, row = row  # 解包iterrows返回的(index, row)
#             img_url = row["source"]
#             json_anno = row["新_结果字段-目标检测标签配置"]
#             img_width = row.get("width")
#             img_height = row.get("height")
#
#             # 生成文件名
#             img_filename = os.path.splitext(os.path.basename(img_url))[0] + ".jpg"
#             label_filename = os.path.splitext(img_filename)[0] + ".txt"
#
#             # 下载图片
#             img_path = os.path.join(img_dir, img_filename)
#             if not download_image(img_url, img_path):
#                 fail += 1
#                 continue
#
#             # 生成标注
#             yolo_anno = json_to_yolo_annotation(json_anno, img_width, img_height)
#             with open(os.path.join(label_dir, label_filename), "w", encoding="utf-8") as f:
#                 f.write(yolo_anno)
#
#             success += 1
#         print(f"{split_name}集：成功 {success} 条 | 失败 {fail} 条")
#
#     process_split(train_df, train_img_dir, train_label_dir, "训练")
#     process_split(val_df, val_img_dir, val_label_dir, "验证")
#
#     # 7. 生成类别映射文件（yolo_classes.txt）
#     class_file = os.path.join(yolo_root, "yolo_classes.txt")
#     with open(class_file, "w", encoding="utf-8") as f:
#         for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
#             f.write(f"{cls_id} {cls_name}\n")
#
#     # 8. 生成YOLO训练所需的yaml文件
#     yaml_data = {
#         "path": os.path.abspath(yolo_root),
#         "train": "train/images",
#         "val": "val/images",
#         "nc": len(class_mapping),
#         "names": [cls_name for cls_name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
#     }
#
#     yaml_path = os.path.join(yolo_root, "dataset.yaml")
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
#
#     # 9. 输出最终结果
#     print(f"\n数据集生成完成！根目录：{os.path.abspath(yolo_root)}")
#     print(f"类别数：{len(class_mapping)} | 类别文件：{class_file}")
#     print(f"YOLO配置文件：{yaml_path}（可直接用于训练）")
#
#
# # # ---------------------- 配置参数 ----------------------
# input_csv_path = "matched_data.csv"  # 替换为你的CSV路径
# val_split = 0.2  # 验证集比例
# class_mapping = {
#     # 示例：手动指定类别ID（可选）
#     # "国槐|生物圈|虫害|国槐尺蠖|严重": 0,
#     # "国槐|生物圈|虫害|国槐尺蠖|明显": 1
# }
# # ------------------------------------------------------
#
# # 安装依赖（若未安装）
# # pip install pandas requests pillow scikit-learn pyyaml tqdm
#
# # 生成数据集
# generate_yolo_dataset(input_csv_path, val_split=val_split, class_mapping=class_mapping)

import pandas as pd
import requests
import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple


# ---------------------- 核心函数：从CSV指定列读取目标标签 ----------------------
def load_target_classes_from_csv(csv_file: str, target_column: str) -> List[str]:
    """
    从CSV文件的指定列读取目标标签，自动去重、过滤空值和无效标签
    :param csv_file: 标签CSV文件路径
    :param target_column: 要读取的标签列名（必须与CSV表头一致）
    :return: 去重后的有效标签列表（为空则保留所有标签）
    """
    # 1. 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"⚠️  警告：标签CSV文件不存在 → {csv_file}")
        print("📌 将保留所有标签（等价于TARGET_CLASSES = []）")
        return []

    try:
        # 2. 读取CSV，仅加载指定列
        df = pd.read_csv(csv_file, encoding="utf-8-sig", usecols=[target_column])

        # 3. 数据清洗：过滤空值、去重、去除空字符串
        target_classes = (
            df[target_column]
            .dropna()  # 过滤NaN值
            .unique()  # 去重
            .tolist()  # 转为列表
        )
        # 进一步过滤空字符串和纯空格标签
        target_classes = [
            str(cls).strip() for cls in target_classes
            if isinstance(cls, (str, int, float)) and str(cls).strip() != ""
        ]

        # 4. 输出结果日志
        if len(target_classes) > 0:
            print(f"✅ 成功从CSV读取标签：")
            print(f"   - 文件路径：{csv_file}")
            print(f"   - 读取列名：{target_column}")
            print(f"   - 有效标签数：{len(target_classes)}")
            print(f"   - 标签列表：{target_classes}")
        else:
            print(f"⚠️  警告：CSV文件 {csv_file} 的 {target_column} 列无有效标签")
            print("📌 将保留所有标签（等价于TARGET_CLASSES = []）")

        return target_classes

    except KeyError:
        print(f"❌ 错误：CSV文件 {csv_file} 中未找到列名 → {target_column}")
        print("📌 将保留所有标签（等价于TARGET_CLASSES = []）")
        return []
    except Exception as e:
        print(f"❌ 错误：读取标签CSV失败 → {str(e)}")
        print("📌 将保留所有标签（等价于TARGET_CLASSES = []）")
        return []


# ---------------------- 步骤1：下载标注数据集（从CSV提取并保存原始数据） ----------------------
def download_annotation_dataset(
        input_csv_path: str,
        raw_data_root: str = "raw_dataset",
        class_mapping: Optional[Dict[str, int]] = None
) -> Tuple[Dict[str, int], str]:
    """
    步骤1：从CSV文件下载图片和生成原始标注，保存到指定文件夹
    核心逻辑：图片存在则跳过下载，不存在才下载
    :param input_csv_path: 输入CSV路径（包含source图片URL、标注字段等）
    :param raw_data_root: 原始数据保存根目录（默认raw_dataset）
    :param class_mapping: 初始类别映射（可选，如{"类别1":0, "类别2":1}）
    :return: (最终类别映射, 原始数据根目录路径)
    """
    # 初始化类别映射
    if class_mapping is None:
        class_mapping = {}
        class_id = 0
    else:
        class_id = max(class_mapping.values(), default=-1) + 1

    # 创建原始数据文件夹（图片+标注）
    raw_img_dir = os.path.join(raw_data_root, "images")
    raw_label_dir = os.path.join(raw_data_root, "labels")
    for dir_path in [raw_img_dir, raw_label_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 图片下载函数（存在则跳过）
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

    # 标注转换函数（生成原始标注，保留所有类别）
    def json_to_yolo_annotation(
            json_str: str,
            img_width: Optional[float],
            img_height: Optional[float]
    ) -> str:
        nonlocal class_mapping, class_id
        yolo_lines = []
        try:
            if pd.isna(json_str) or not isinstance(json_str, str):
                return ""
            data = json.loads(json_str)
            objects = data.get("objects", [])
            img_width = img_width or data.get("width", 1)
            img_height = img_height or data.get("height", 1)

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                # 处理类别ID（保留所有原始类别）
                obj_name = obj.get("name", "unknown")
                if obj_name not in class_mapping:
                    class_mapping[obj_name] = class_id
                    class_id += 1
                cid = class_mapping[obj_name]

                # 处理坐标（YOLO格式：归一化中心坐标+宽高）
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
            print(f"\n❌ JSON解析失败（前50字符）：{str(json_str)[:50]}...")
        except Exception as e:
            print(f"\n❌ 标注转换失败：{e}")
        return "\n".join(yolo_lines)

    # 读取CSV并处理
    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
        print(f"步骤1：成功读取CSV，共 {len(df)} 行数据")
    except Exception as e:
        raise Exception(f"步骤1：CSV读取失败：{e}") from e

    # 批量下载图片和生成标注
    success_count = 0
    fail_count = 0
    skip_count = 0
    for idx, row in tqdm(enumerate(df.iterrows()), total=len(df), desc="步骤1：下载并生成原始数据", unit="条"):
        _, row = row
        img_url = row["source"]
        json_anno = row["新_结果字段-目标检测标签配置"]
        img_width = row.get("width")
        img_height = row.get("height")

        # 生成文件名（处理特殊字符）
        img_basename = os.path.splitext(os.path.basename(img_url))[0]
        img_basename = img_basename.replace("?", "").replace("&", "_").replace("/", "_")
        img_filename = f"{img_basename}.jpg"
        label_filename = f"{img_basename}.txt"

        # 下载图片（自动判断是否存在）
        img_path = os.path.join(raw_img_dir, img_filename)
        download_result = download_image(img_url, img_path)
        if not download_result:
            fail_count += 1
            continue
        if os.path.exists(img_path):
            skip_count += 1

        # 生成原始标注
        yolo_anno = json_to_yolo_annotation(json_anno, img_width, img_height)
        label_path = os.path.join(raw_label_dir, label_filename)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(yolo_anno)

        success_count += 1

    # 输出步骤1结果
    print(f"\n步骤1：原始数据生成完成！")
    print(f" - 总处理数据：{len(df)} 条")
    print(f" - 成功（下载/跳过+标注）：{success_count} 条")
    print(f" - 下载失败：{fail_count} 条")
    print(f" - 跳过已存在图片：{skip_count - (success_count - fail_count)} 条")
    print(f" - 原始图片目录：{raw_img_dir}")
    print(f" - 原始标注目录：{raw_label_dir}")
    print(f" - 识别到的所有类别：{sorted(class_mapping.items(), key=lambda x: x[1])}")

    return class_mapping, raw_data_root


# ---------------------- 步骤2：筛选数据集（保留指定标签，删除其他标注） ----------------------
def filter_dataset(
        raw_data_root: str,
        filtered_data_root: str = "filtered_dataset",
        target_classes: List[str] = None,
        class_mapping: Dict[str, int] = None
) -> str:
    """
    步骤2：筛选符合要求的数据集（保留指定标签，删除其他标注信息）
    核心修改1：只保留目标标签中在数据集中实际存在的标签，过滤无效标签
    核心修改2：每次筛选前清空filtered_dataset目录的旧数据，只保留新数据
    逻辑：当target_classes为空（或None/空列表）时，保留所有标签（不筛选）
    :param raw_data_root: 步骤1生成的原始数据根目录
    :param filtered_data_root: 筛选后数据保存目录（默认filtered_dataset）
    :param target_classes: 需保留的目标标签列表（为空则保留所有标签）
    :param class_mapping: 步骤1生成的类别映射（必须传入）
    :return: 筛选后数据根目录路径
    """
    # 校验class_mapping
    if class_mapping is None or len(class_mapping) == 0:
        raise ValueError("步骤2：class_mapping不能为空，请传入步骤1生成的类别映射")

    # ---------------------- 新增：清空筛选目录的旧数据 ----------------------
    filtered_img_dir = os.path.join(filtered_data_root, "images")
    filtered_label_dir = os.path.join(filtered_data_root, "labels")
    # 若目录已存在，删除所有内容（包括子文件和子目录）
    if os.path.exists(filtered_data_root):
        print(f"🗑️  清空旧筛选数据：{filtered_data_root}")
        # 删除整个目录（包括内容），然后重新创建空目录
        shutil.rmtree(filtered_data_root)
    # 重新创建空的筛选目录
    for dir_path in [filtered_img_dir, filtered_label_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------------------------

    # 处理target_classes：只保留数据集中存在的标签
    dataset_existing_classes = set(class_mapping.keys())  # 数据集中实际存在的所有标签
    if target_classes is None or len(target_classes) == 0:
        print("步骤2：target_classes为空，将保留所有标签，不进行筛选")
        target_class_ids = list(class_mapping.values())
        target_classes = list(class_mapping.keys())
    else:
        # 核心修改：筛选出"目标标签"和"数据集存在标签"的交集
        target_classes = [cls for cls in target_classes if cls in dataset_existing_classes]
        if len(target_classes) > 0:
            print(f"✅ 筛选出数据集中实际存在的目标标签：{target_classes}")
            target_class_ids = [class_mapping[cls] for cls in target_classes]
        else:
            print("⚠️  警告：所有目标标签在数据集中均不存在，将保留所有标签")
            target_class_ids = list(class_mapping.values())
            target_classes = list(class_mapping.keys())

    # 获取原始数据的所有图片和标注文件
    raw_img_dir = os.path.join(raw_data_root, "images")
    raw_label_dir = os.path.join(raw_data_root, "labels")
    if not os.path.exists(raw_img_dir) or not os.path.exists(raw_label_dir):
        raise FileNotFoundError(f"步骤2：原始数据目录不存在：{raw_img_dir} 或 {raw_label_dir}")

    # 筛选逻辑：保留目标标签（或所有标签）
    success_count = 0
    no_target_count = 0
    label_files = list(Path(raw_label_dir).glob("*.txt"))

    for label_file in tqdm(label_files, desc="步骤2：筛选数据集", unit="个"):
        label_basename = label_file.stem
        img_filename = f"{label_basename}.jpg"
        raw_img_path = os.path.join(raw_img_dir, img_filename)
        filtered_img_path = os.path.join(filtered_img_dir, img_filename)
        filtered_label_path = os.path.join(filtered_label_dir, label_file.name)

        # 跳过不存在的图片
        if not os.path.exists(raw_img_path):
            print(f"\n步骤2：图片不存在，跳过：{raw_img_path}")
            continue

        # 读取原始标注，筛选目标标签
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            if cid in target_class_ids:
                filtered_lines.append(line)

        # 保留有有效标注的文件
        if len(filtered_lines) > 0:
            shutil.copy2(raw_img_path, filtered_img_path)
            with open(filtered_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(filtered_lines))
            success_count += 1
        else:
            no_target_count += 1

    # 输出步骤2结果
    print(f"\n步骤2：数据集筛选完成！")
    print(f" - 保留的标签：{target_classes}（对应ID：{target_class_ids}）")
    print(f" - 保留有效数据：{success_count} 条")
    print(f" - 无有效标注被过滤：{no_target_count} 条")
    print(f" - 筛选后图片目录：{filtered_img_dir}")
    print(f" - 筛选后标注目录：{filtered_label_dir}")

    return filtered_data_root


# ---------------------- 步骤3：划分数据集（生成YOLOv11训练格式） ----------------------
def split_yolov11_dataset(
        filtered_data_root: str,
        yolo_root: str = "yolo_dataset",
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        target_classes: List[str] = None,
        class_mapping: Dict[str, int] = None
) -> str:
    """
    步骤3：划分训练集/验证集/测试集，生成YOLOv11所需的数据集结构和配置文件
    核心修改：每次划分前清空yolo_dataset目录的旧数据，只保留新数据
    :param filtered_data_root: 步骤2生成的筛选后数据根目录
    :param yolo_root: YOLOv11数据集根目录（默认yolo_dataset）
    :param train_split: 训练集比例（默认0.7）
    :param val_split: 验证集比例（默认0.2）
    :param test_split: 测试集比例（默认0.1）
    :param target_classes: 保留的目标标签列表（为空则使用所有类别）
    :param class_mapping: 步骤1生成的类别映射
    :return: YOLOv11数据集根目录路径
    """
    # 验证比例之和为1
    if not abs(train_split + val_split + test_split - 1.0) < 1e-6:
        raise ValueError(f"步骤3：数据集比例之和必须为1，当前：{train_split}+{val_split}+{test_split}")
    if class_mapping is None or len(class_mapping) == 0:
        raise ValueError("步骤3：class_mapping不能为空，请传入步骤1生成的类别映射")

    # ---------------------- 新增：清空YOLO数据集目录的旧数据 ----------------------
    if os.path.exists(yolo_root):
        print(f"🗑️  清空旧YOLO数据集：{yolo_root}")
        shutil.rmtree(yolo_root)
    # --------------------------------------------------------------------------

    # 处理target_classes：只保留数据集中存在的标签（与步骤2保持一致）
    dataset_existing_classes = set(class_mapping.keys())
    if target_classes is None or len(target_classes) == 0:
        target_classes = list(class_mapping.keys())
        print(f"步骤3：target_classes为空，使用所有类别：{target_classes}")
    else:
        target_classes = [cls for cls in target_classes if cls in dataset_existing_classes]
        if len(target_classes) == 0:
            target_classes = list(class_mapping.keys())
            print(f"步骤3：所有目标标签在数据集中均不存在，使用所有类别：{target_classes}")
        else:
            print(f"步骤3：使用数据集中存在的目标标签：{target_classes}")

    # 提取筛选后的图片和标注文件
    filtered_img_dir = os.path.join(filtered_data_root, "images")
    filtered_label_dir = os.path.join(filtered_data_root, "labels")
    if not os.path.exists(filtered_img_dir) or not os.path.exists(filtered_label_dir):
        raise FileNotFoundError(f"步骤3：筛选后数据目录不存在：{filtered_img_dir} 或 {filtered_label_dir}")

    # 获取所有有效数据（图片+标注成对存在）
    data_files = []
    img_files = list(Path(filtered_img_dir).glob("*.jpg"))
    for img_file in img_files:
        img_basename = img_file.stem
        label_file = Path(filtered_label_dir) / f"{img_basename}.txt"
        if label_file.exists():
            data_files.append((str(img_file), str(label_file)))

    if len(data_files) == 0:
        raise ValueError("步骤3：筛选后的数据为空，无法划分数据集")

    print(f"步骤3：共获取 {len(data_files)} 条有效数据，开始划分...")

    # 创建YOLOv11数据集文件夹结构
    train_img_dir = os.path.join(yolo_root, "train", "images")
    train_label_dir = os.path.join(yolo_root, "train", "labels")
    val_img_dir = os.path.join(yolo_root, "val", "images")
    val_label_dir = os.path.join(yolo_root, "val", "labels")
    test_img_dir = os.path.join(yolo_root, "test", "images")
    test_label_dir = os.path.join(yolo_root, "test", "labels")

    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 划分数据集
    train_val_files, test_files = train_test_split(data_files, test_size=test_split, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=val_split / (train_split + val_split),
                                              random_state=42)

    # 复制文件到对应目录
    def copy_files(files: List[tuple], img_dir: str, label_dir: str, split_name: str):
        for img_path, label_path in tqdm(files, desc=f"步骤3：复制{split_name}集", unit="条"):
            img_dst = os.path.join(img_dir, os.path.basename(img_path))
            label_dst = os.path.join(label_dir, os.path.basename(label_path))
            shutil.copy2(img_path, img_dst)
            shutil.copy2(label_path, label_dst)

    copy_files(train_files, train_img_dir, train_label_dir, "训练")
    copy_files(val_files, val_img_dir, val_label_dir, "验证")
    copy_files(test_files, test_img_dir, test_label_dir, "测试")

    # 生成YOLOv11所需的类别文件和yaml配置文件
    target_class_mapping = {cls: class_mapping[cls] for cls in target_classes}
    sorted_target_classes = sorted(target_class_mapping.items(), key=lambda x: x[1])
    class_file = os.path.join(yolo_root, "yolo_classes.txt")
    with open(class_file, "w", encoding="utf-8") as f:
        for cls_name, cls_id in sorted_target_classes:
            f.write(f"{cls_id} {cls_name}\n")

    # 生成dataset.yaml
    yaml_data = {
        "path": os.path.abspath(yolo_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(target_classes),
        "names": [cls_name for cls_name, _ in sorted_target_classes]
    }
    yaml_path = os.path.join(yolo_root, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    # 输出步骤3结果
    print(f"\n步骤3：YOLOv11数据集生成完成！")
    print(f" - 数据集根目录：{os.path.abspath(yolo_root)}")
    print(f" - 训练集：{len(train_files)} 条 | 验证集：{len(val_files)} 条 | 测试集：{len(test_files)} 条")
    print(f" - 类别数：{len(target_classes)} | 类别文件：{class_file}")
    print(f" - YOLO配置文件：{yaml_path}（可直接用于YOLOv11训练）")

    return yolo_root


# 说明：
# 上面的函数原本在 process_step.py 中带有 CLI 示例入口。
# 合并为单文件后，Streamlit 运行时 __name__ == "__main__"，会误触发该示例。
# 为避免无关报错（例如 missing matched_data.csv），已移除示例入口。


st.set_page_config(page_title="YOLO 数据处理流水线", layout="wide")


def inject_style():
    st.markdown(
        """
<style>
:root {
  --bg: #f5f7fb;
  --bg-2: #eef2f7;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --accent-2: #60a5fa;
  --accent-3: #38bdf8;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
}

.stApp {
  background:
    radial-gradient(900px circle at 15% 10%, rgba(96, 165, 250, 0.2), transparent 55%),
    radial-gradient(800px circle at 85% 0%, rgba(56, 189, 248, 0.18), transparent 55%),
    linear-gradient(180deg, #f6f8fc 0%, #eef2f7 45%, #f5f7fb 100%);
  color: var(--text);
  font-family: "Avenir Next", "Source Sans Pro", "Noto Sans", sans-serif;
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(59, 130, 246, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59, 130, 246, 0.08) 1px, transparent 1px);
  background-size: 36px 36px;
  pointer-events: none;
  z-index: 0;
}

.hero-title {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, #3b82f6, #60a5fa, #2563eb, #3b82f6);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: titleGlow 6s linear infinite;
  text-shadow: 0 0 18px rgba(59, 130, 246, 0.25);
}

@keyframes titleGlow {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}

.glow-frame {
  position: relative;
  border-radius: 16px;
  padding: 2px;
  background: linear-gradient(120deg, rgba(59, 130, 246, 0.18), rgba(37, 99, 235, 0.25), rgba(59, 130, 246, 0.18));
  background-size: 200% 200%;
  animation: borderFlow 8s ease infinite;
}

.glow-frame > .glow-inner {
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.92);
  padding: 14px 16px;
}

@keyframes borderFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.busy-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: #3b82f6;
}

.busy-dots span {
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-left: 3px;
  border-radius: 50%;
  background: #3b82f6;
  animation: pulse 1.2s infinite ease-in-out;
}

.busy-dots span:nth-child(2) { animation-delay: 0.2s; }
.busy-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border-right: 1px solid var(--border);
}

.sidebar-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.5rem;
}

.panel {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.12), #ffffff);
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

.kpi {
  font-size: 0.85rem;
  color: var(--muted);
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.chip-wait {
  background: #f1f5f9;
  color: #64748b;
  border-color: #e2e8f0;
}

.chip-done {
  background: #dcfce7;
  color: #166534;
  border-color: #86efac;
}

.chip-skip {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}

.file-card {
  background: #f8fafc;
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 12px;
  padding: 10px 12px;
  min-height: 70px;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.06);
}

.file-name {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text);
  word-break: break-all;
}

.file-meta {
  font-size: 0.75rem;
  color: var(--muted);
}

button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
  border: none;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
}

div[data-baseweb="input"] > div,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: rgba(37, 99, 235, 0.2) !important;
}

div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: rgba(100, 116, 139, 0.8) !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 12px;
  overflow: hidden;
}

div[data-testid="stExpander"] {
  border-radius: 14px;
  border: 1px solid rgba(37, 99, 235, 0.12);
  background: #ffffff;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

div[data-testid="stExpander"] > details > summary {
  padding: 0.4rem 1rem;
  font-weight: 600;
  color: var(--text);
}

.stepper {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.step {
  display: flex;
  align-items: center;
  gap: 8px;
}

.step-circle {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  border: 1px solid rgba(37, 99, 235, 0.2);
  background: #ffffff;
  color: #64748b;
}

.step-circle.done {
  background: rgba(34, 197, 94, 0.2);
  color: #86efac;
  border-color: rgba(34, 197, 94, 0.5);
}

.step-circle.active {
  background: #dbeafe;
  color: #1d4ed8;
  border-color: #93c5fd;
}

.step-circle.locked {
  background: #f1f5f9;
  color: #94a3b8;
}

.step-circle.skipped {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}

.step-label {
  font-size: 0.8rem;
  color: var(--text);
  font-weight: 600;
}

.step-line {
  flex: 1;
  height: 2px;
  min-width: 24px;
  background: #e2e8f0;
}

.step-line.line-done {
  background: linear-gradient(90deg, #22c55e, #16a34a);
}

.step-line.line-skip {
  background: repeating-linear-gradient(90deg, #f59e0b 0 6px, #fde68a 6px 12px);
}

.step-line.line-lock {
  background: #e2e8f0;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 8px;
}

.stat-card {
  background: #ffffff;
  border: 1px solid rgba(37, 99, 235, 0.12);
  border-radius: 12px;
  padding: 12px;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.05);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--muted);
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 4px;
  color: var(--text);
}

.stat-hint {
  font-size: 0.7rem;
  color: var(--muted);
  margin-top: 2px;
}

.dependency-card {
  background: #ffffff;
  border: 1px solid rgba(37, 99, 235, 0.12);
  border-radius: 14px;
  padding: 12px 16px;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

.file-manager {
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 14px;
  padding: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.05);
}

.fm-node {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 8px;
}

.fm-selected {
  background: rgba(37, 99, 235, 0.12);
  border: 1px solid rgba(37, 99, 235, 0.3);
}

.fm-node:hover {
  background: rgba(59, 130, 246, 0.08);
}

.fm-icon {
  width: 18px;
  height: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.fm-icon svg {
  width: 18px;
  height: 14px;
  fill: currentColor;
  color: #2563eb;
}

.fm-depth-0 .fm-icon svg { color: #0284c7; }
.fm-depth-1 .fm-icon svg { color: #2563eb; }
.fm-depth-2 .fm-icon svg { color: #7c3aed; }
.fm-depth-3 .fm-icon svg { color: #059669; }
.fm-depth-4 .fm-icon svg { color: #f59e0b; }

.fm-name {
  font-size: 0.82rem;
  color: #0f172a;
  font-weight: 600;
}

.fm-path {
  font-size: 0.75rem;
  color: #64748b;
}

.drop-zone {
  border: 1px dashed rgba(37, 99, 235, 0.4);
  border-radius: 12px;
  padding: 10px;
  text-align: center;
  color: #2563eb;
  background: rgba(59, 130, 246, 0.05);
  font-size: 0.8rem;
}

hr {
  border: none;
  border-top: 1px solid rgba(37, 99, 235, 0.12);
}
</style>
        """,
        unsafe_allow_html=True,
    )


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


def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def clear_output_root(output_root: Path, keep_inputs: bool = True, keep_files=None):
    keep = set(keep_files or [])
    if keep_inputs:
        keep.add("input_csvs")
    for item in output_root.iterdir() if output_root.exists() else []:
        if item.name in keep:
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception:
            pass


def show_confirm_dialog(state_key: str, title: str, body: str, on_confirm):
    def _handle_confirm():
        on_confirm()
        st.session_state[state_key] = False
        trigger_rerun()

    def _handle_cancel():
        st.session_state[state_key] = False
        trigger_rerun()

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dialog():
            st.write(body)
            col1, col2 = st.columns(2)
            if col1.button("确认删除", key=f"{state_key}_confirm", width='stretch'):
                _handle_confirm()
            if col2.button("取消", key=f"{state_key}_cancel", width='stretch'):
                _handle_cancel()
        _dialog()
    else:
        st.warning(body)
        col1, col2 = st.columns(2)
        if col1.button("确认删除", key=f"{state_key}_confirm", width='stretch'):
            _handle_confirm()
        if col2.button("取消", key=f"{state_key}_cancel", width='stretch'):
            _handle_cancel()


def build_train_template_payload():
    saved_data_yaml = st.session_state.get("train_dataset_manual") or st.session_state.get("train_dataset_choice") or ""
    return {
        "dataset_root": st.session_state.get("train_dataset_root"),
        "data_yaml": saved_data_yaml,
        "model_path": st.session_state.get("train_model_path"),
        "project": st.session_state.get("train_project_input"),
        "name": st.session_state.get("train_name_input"),
        "exist_ok": st.session_state.get("train_exist_ok"),
        "epochs": st.session_state.get("train_epochs"),
        "imgsz": st.session_state.get("train_imgsz"),
        "batch": st.session_state.get("train_batch"),
        "workers": st.session_state.get("train_workers"),
        "device": st.session_state.get("train_device"),
        "amp": st.session_state.get("train_amp"),
        "cache": st.session_state.get("train_cache"),
        "resume": st.session_state.get("train_resume"),
        "optimizer": st.session_state.get("train_optimizer"),
        "seed": st.session_state.get("train_seed"),
        "patience": st.session_state.get("train_patience"),
        "cos_lr": st.session_state.get("train_cos_lr"),
        "close_mosaic": st.session_state.get("train_close_mosaic"),
        "save_period": st.session_state.get("train_save_period"),
        "advanced_text": st.session_state.get("train_advanced"),
        "cuda_visible_devices": st.session_state.get("train_cuda_visible"),
        "scan_yaml": st.session_state.get("train_scan_yaml"),
    }


def save_template_file(target: Path, payload: dict):
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def check_train_dependencies():
    missing = []
    if importlib.util.find_spec("ultralytics") is None:
        missing.append("ultralytics")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    return missing


@st.cache_data(show_spinner=False)
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


def format_int_safe(value):
    return "-" if value is None else f"{value:,}"


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


@st.cache_data(show_spinner=False)
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


def build_tree_flat(root: Path, include_hidden: bool, max_depth: int, max_nodes: int):
    nodes = []
    root_str = str(root)

    def _walk(path: Path, parent: str, depth: int):
        if depth > max_depth or len(nodes) >= max_nodes:
            return
        try:
            children = [
                p for p in path.iterdir()
                if p.is_dir() and (include_hidden or not p.name.startswith("."))
            ]
        except Exception:
            return
        children = sorted(children, key=lambda x: x.name.lower())
        for child in children:
            if len(nodes) >= max_nodes:
                break
            child_str = str(child)
            try:
                has_children = any(
                    p.is_dir() and (include_hidden or not p.name.startswith("."))
                    for p in child.iterdir()
                )
            except Exception:
                has_children = False
            nodes.append({
                "id": child_str,
                "parent": parent,
                "name": child.name,
                "path": child_str,
                "depth": depth,
                "has_children": has_children,
            })
            _walk(child, child_str, depth + 1)

    _walk(root, root_str, 0)
    return nodes, root_str


def filter_tree_nodes(nodes, query: str):
    if not query:
        return nodes
    key = query.lower()
    by_id = {n["id"]: n for n in nodes}
    parent_map = {n["id"]: n["parent"] for n in nodes}
    keep = set()
    for n in nodes:
        if key in n["name"].lower():
            current = n["id"]
            while current and current in parent_map:
                keep.add(current)
                current = parent_map.get(current)
    return [n for n in nodes if n["id"] in keep]


def render_advanced_tree_component(nodes, root_id: str, expanded, selected):
    payload = json.dumps({
        "nodes": nodes,
        "root": root_id,
        "expanded": expanded or [],
        "selected": selected or "",
    }, ensure_ascii=False)
    html = f"""
    <style>
      .adv-tree {{ font-family: inherit; font-size: 13px; color: #0f172a; }}
      .adv-row {{ display: flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 6px; }}
      .adv-row:hover {{ background: rgba(59,130,246,0.08); }}
      .adv-selected {{ background: rgba(37,99,235,0.12); border: 1px solid rgba(37,99,235,0.3); }}
      .adv-toggle {{ width: 14px; text-align: center; cursor: pointer; }}
      .adv-name {{ font-weight: 600; }}
      .adv-icon svg {{ width: 16px; height: 12px; color: #2563eb; }}
      .adv-menu {{ position: fixed; background: #fff; border: 1px solid rgba(15,23,42,0.15); box-shadow: 0 10px 24px rgba(15,23,42,0.15); border-radius: 8px; padding: 6px; display: none; z-index: 9999; }}
      .adv-menu button {{ width: 100%; margin: 2px 0; padding: 6px 10px; border-radius: 6px; border: 1px solid #e2e8f0; background: #f8fafc; cursor: pointer; }}
    </style>
    <div class="adv-tree" id="adv-tree"></div>
    <div class="adv-menu" id="adv-menu">
      <button data-action="preview">预览</button>
      <button data-action="set_root">设为根目录</button>
      <button data-action="copy">复制路径</button>
    </div>
    <script>
      const payload = {payload};
      const nodes = payload.nodes || [];
      const rootId = payload.root;
      const expanded = new Set(payload.expanded || []);
      let selected = payload.selected || '';
      const treeEl = document.getElementById('adv-tree');
      const menuEl = document.getElementById('adv-menu');
      let menuPath = '';

      const byParent = {{}};
      nodes.forEach(n => {{
        if (!byParent[n.parent]) byParent[n.parent] = [];
        byParent[n.parent].push(n);
      }});

      function send(action, path) {{
        const msg = {{
          action,
          path,
          expanded: Array.from(expanded),
          selected: path
        }};
        window.parent.postMessage({{
          isStreamlitMessage: true,
          type: 'streamlit:setComponentValue',
          value: JSON.stringify(msg)
        }}, '*');
      }}

      function render(parentId, container, depth) {{
        const children = byParent[parentId] || [];
        children.forEach(node => {{
          const row = document.createElement('div');
          row.className = 'adv-row' + (node.path === selected ? ' adv-selected' : '');
          row.style.marginLeft = (depth * 14) + 'px';
          const toggle = document.createElement('div');
          toggle.className = 'adv-toggle';
          toggle.textContent = node.has_children ? (expanded.has(node.path) ? '▾' : '▸') : '';
          toggle.onclick = (e) => {{
            e.stopPropagation();
            if (!node.has_children) return;
            if (expanded.has(node.path)) expanded.delete(node.path);
            else expanded.add(node.path);
            renderTree();
          }};
          const icon = document.createElement('span');
          icon.className = 'adv-icon';
          icon.innerHTML = "<svg viewBox='0 0 24 16'><path d='M2 3.5C2 2.7 2.7 2 3.5 2h5.2c.4 0 .8.2 1 .5l.9 1.3c.2.3.6.5 1 .5h8.9c.8 0 1.5.7 1.5 1.5v6.2c0 .8-.7 1.5-1.5 1.5H3.5c-.8 0-1.5-.7-1.5-1.5V3.5z'/></svg>";
          const name = document.createElement('span');
          name.className = 'adv-name';
          name.textContent = node.name;
          row.appendChild(toggle);
          row.appendChild(icon);
          row.appendChild(name);
          row.onclick = () => {{
            selected = node.path;
            send('select', node.path);
          }};
          row.oncontextmenu = (e) => {{
            e.preventDefault();
            menuPath = node.path;
            menuEl.style.display = 'block';
            menuEl.style.left = e.clientX + 'px';
            menuEl.style.top = e.clientY + 'px';
          }};
          container.appendChild(row);
          if (node.has_children && expanded.has(node.path)) {{
            render(node.path, container, depth + 1);
          }}
        }});
      }}

      function renderTree() {{
        treeEl.innerHTML = '';
        render(rootId, treeEl, 0);
      }}

      document.addEventListener('click', (e) => {{
        if (!menuEl.contains(e.target)) {{
          menuEl.style.display = 'none';
        }}
      }});

      menuEl.addEventListener('click', (e) => {{
        const action = e.target.getAttribute('data-action');
        if (!action) return;
        if (action === 'copy') {{
          try {{ navigator.clipboard.writeText(menuPath); }} catch (err) {{}}
        }} else {{
          send(action, menuPath);
        }}
        menuEl.style.display = 'none';
      }});

      renderTree();
    </script>
    """
    return components.html(html, height=520, scrolling=True)

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


def ensure_favorite_groups(default_group: str = "默认"):
    if "train_favorite_groups" not in st.session_state:
        groups = {}
        legacy = st.session_state.get("train_favorite_paths", [])
        groups[default_group] = list(legacy) if legacy else []
        st.session_state["train_favorite_groups"] = groups
    return st.session_state.get("train_favorite_groups", {})


def add_favorite_path(path_str: str, group: str = "默认", max_items: int = 12):
    if not path_str:
        return
    groups = ensure_favorite_groups()
    if group not in groups:
        groups[group] = []
    if path_str in groups[group]:
        return
    groups[group].append(path_str)
    groups[group] = groups[group][:max_items]
    st.session_state["train_favorite_groups"] = groups


def remove_favorite_path(path_str: str, group: str):
    groups = ensure_favorite_groups()
    if group in groups and path_str in groups[group]:
        groups[group].remove(path_str)
        st.session_state["train_favorite_groups"] = groups


def add_favorite_group(name: str):
    groups = ensure_favorite_groups()
    if not name:
        return
    if name not in groups:
        groups[name] = []
        st.session_state["train_favorite_groups"] = groups


def delete_favorite_group(name: str, default_group: str = "默认"):
    groups = ensure_favorite_groups()
    if name in groups and name != default_group:
        del groups[name]
        st.session_state["train_favorite_groups"] = groups


def build_category_preview_options(dataset_root: Path, dataset_yaml_paths):
    options = {}
    if not dataset_yaml_paths:
        return options
    for path in dataset_yaml_paths:
        try:
            root = Path(dataset_root)
            label = str(Path(path).parent.name)
            if root.exists():
                try:
                    rel = Path(path).parent.relative_to(root)
                    label = str(rel)
                except Exception:
                    pass
            if label in options:
                label = f"{label} ({Path(path).parent})"
            options[label] = str(path)
        except Exception:
            continue
    return options


def safe_filename(value: str) -> str:
    if not value:
        return "train"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "train"


def build_dir_tree_nodes(root: Path, max_depth: int = 4, max_nodes: int = 2000, include_hidden: bool = False):
    count = 0

    def _label(path: Path) -> str:
        name = path.name or str(path)
        return f"DIR {name}"

    def _children(path: Path, depth: int):
        nonlocal count
        if depth > max_depth or count >= max_nodes:
            return []
        try:
            entries = [
                p for p in path.iterdir()
                if p.is_dir() and (include_hidden or not p.name.startswith("."))
            ]
        except Exception:
            return []
        entries = sorted(entries, key=lambda x: x.name.lower())
        nodes = []
        for p in entries:
            if count >= max_nodes:
                break
            count += 1
            node = {"label": _label(p), "value": str(p)}
            children = _children(p, depth + 1)
            if children:
                node["children"] = children
            nodes.append(node)
        return nodes

    if not root.exists() or not root.is_dir():
        return [], 0
    root_node = {"label": _label(root), "value": str(root)}
    root_children = _children(root, 1)
    if root_children:
        root_node["children"] = root_children
    return [root_node], count


@st.cache_data(show_spinner=False)
def build_dir_tree_nodes_cached(root_str: str, max_depth: int, max_nodes: int, include_hidden: bool):
    return build_dir_tree_nodes(Path(root_str), max_depth=max_depth, max_nodes=max_nodes, include_hidden=include_hidden)


def render_copy_button(path_value: str, key: str):
    escaped = path_value.replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <button style="padding:4px 10px;border-radius:8px;border:1px solid #cbd5f5;background:#f8fafc;cursor:pointer;" onclick="navigator.clipboard.writeText('{escaped}')">复制路径</button>
    """
    components.html(html, height=32, width=110)


def copy_to_clipboard(path_value: str):
    escaped = json.dumps(path_value)
    html = f"""
    <script>
    try {{
      navigator.clipboard.writeText({escaped});
    }} catch (e) {{}}
    </script>
    """
    components.html(html, height=0, width=0)


def add_recent_path(path_str: str, max_items: int = 8):
    if not path_str:
        return
    recent = list(st.session_state.get("train_recent_paths", []))
    if path_str in recent:
        recent.remove(path_str)
    recent.insert(0, path_str)
    st.session_state["train_recent_paths"] = recent[:max_items]


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


def _stable_key(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


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


def render_icon_tree_custom(root: Path, include_hidden: bool, max_depth: int, max_nodes: int, filter_query: str = ""):
    expanded = set(st.session_state.get("train_icon_tree_expanded", []))
    selected = st.session_state.get("train_icon_tree_selected", "")
    menu_open = st.session_state.get("train_icon_tree_menu_open", "")
    counter = [0]
    filter_key = filter_query.strip().lower()

    def _matches(path: Path) -> bool:
        if not filter_key:
            return True
        return filter_key in path.name.lower()

    def _has_match_descendant(path: Path, depth: int) -> bool:
        if depth > max_depth:
            return False
        try:
            for child in path.iterdir():
                if child.is_dir() and (include_hidden or not child.name.startswith(".")):
                    if _matches(child):
                        return True
                    if _has_match_descendant(child, depth + 1):
                        return True
        except Exception:
            return False
        return False

    def _render(path: Path, depth: int):
        if depth > max_depth or counter[0] >= max_nodes:
            return
        try:
            children = [
                p for p in path.iterdir()
                if p.is_dir() and (include_hidden or not p.name.startswith("."))
            ]
        except Exception:
            return
        children = sorted(children, key=lambda x: x.name.lower())
        for child in children:
            if counter[0] >= max_nodes:
                break
            counter[0] += 1
            child_str = str(child)
            if filter_key and not _matches(child) and not _has_match_descendant(child, depth + 1):
                continue
            is_expanded = child_str in expanded
            selected_class = "fm-selected" if child_str == selected else ""
            indent_px = depth * 16

            cols = st.columns([0.8, 5.4, 1.2, 1.2])
            with cols[0]:
                toggle_label = "▾" if is_expanded else "▸"
                if st.button(toggle_label, key=f"train_icon_toggle_{_stable_key(child_str)}"):
                    if is_expanded:
                        expanded.discard(child_str)
                    else:
                        expanded.add(child_str)
                    st.session_state["train_icon_tree_expanded"] = list(expanded)
                    trigger_rerun()
            with cols[1]:
                drag_path = child_str.replace("\\", "\\\\").replace("'", "\\'")
                st.markdown(
                    (
                        f"<div class='fm-node fm-depth-{depth} {selected_class}' "
                        f"style='margin-left:{indent_px}px;' "
                        f"draggable='true' "
                        f"ondragstart=\"event.dataTransfer.setData('text/plain', '{drag_path}');\">"
                        "<span class='fm-icon'>"
                        "<svg viewBox='0 0 24 16' aria-hidden='true'>"
                        "<path d='M2 3.5C2 2.7 2.7 2 3.5 2h5.2c.4 0 .8.2 1 .5l.9 1.3c.2.3.6.5 1 .5h8.9c.8 0 1.5.7 1.5 1.5v6.2c0 .8-.7 1.5-1.5 1.5H3.5c-.8 0-1.5-.7-1.5-1.5V3.5z'/>"
                        "</svg>"
                        "</span>"
                        f"<span class='fm-name'>{child.name}</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            with cols[2]:
                if st.button("选择", key=f"train_icon_pick_{_stable_key(child_str)}", width='stretch'):
                    st.session_state["train_icon_tree_selected"] = child_str
                    st.session_state["train_preview_path"] = child_str
                    add_recent_path(child_str)
                    st.session_state["train_dataset_root"] = child_str
                    trigger_rerun()
            with cols[3]:
                if st.button("⋯", key=f"train_icon_menu_{_stable_key(child_str)}", width='stretch'):
                    st.session_state["train_icon_tree_menu_open"] = "" if menu_open == child_str else child_str
                    trigger_rerun()

            if menu_open == child_str:
                action_cols = st.columns([1.4, 1.8, 1.4])
                with action_cols[0]:
                    if st.button("预览", key=f"train_icon_preview_{_stable_key(child_str)}", width='stretch'):
                        st.session_state["train_preview_path"] = child_str
                        trigger_rerun()
                with action_cols[1]:
                    if st.button("设为根目录", key=f"train_icon_root_{_stable_key(child_str)}", width='stretch'):
                        st.session_state["train_dataset_root"] = child_str
                        st.session_state["train_browse_root"] = child_str
                        add_recent_path(child_str)
                        trigger_rerun()
                with action_cols[2]:
                    render_copy_button(child_str, f"train_icon_copy_{_stable_key(child_str)}")

            if is_expanded:
                _render(child, depth + 1)

    _render(root, 0)


def format_bytes(value: int):
    if value is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"


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


def run_yolo_training(model_path: str, data_yaml: str, train_kwargs: dict, env_vars: dict):
    buffer = io.StringIO()
    model = None
    error = None
    results = None
    save_dir = None
    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            if env_vars:
                for key, value in env_vars.items():
                    if value:
                        os.environ[str(key)] = str(value)
            from ultralytics import YOLO

            model = YOLO(model_path)
            results = model.train(data=data_yaml, **train_kwargs)
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None) if trainer else None
            if save_dir is None and hasattr(results, "save_dir"):
                save_dir = getattr(results, "save_dir")
        except Exception as exc:
            error = exc
    return results, buffer.getvalue(), save_dir, error


LOG_DONE = object()


class StreamQueueWriter:
    def __init__(self, log_queue: "queue.Queue[str]"):
        self.log_queue = log_queue
        self._buffer = ""

    def write(self, data):
        if not data:
            return
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log_queue.put(line)

    def flush(self):
        if self._buffer:
            self.log_queue.put(self._buffer)
            self._buffer = ""


def _extract_epoch_info(line: str):
    if not line:
        return None
    match = re.search(r"[Ee]poch\s*(\d+)\s*/\s*(\d+)", line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def run_yolo_training_stream(model_path: str, data_yaml: str, train_kwargs: dict, env_vars: dict, log_queue: "queue.Queue", result_holder: dict):
    writer = StreamQueueWriter(log_queue)
    with redirect_stdout(writer), redirect_stderr(writer):
        try:
            if env_vars:
                for key, value in env_vars.items():
                    if value:
                        os.environ[str(key)] = str(value)
            from ultralytics import YOLO

            model = YOLO(model_path)
            results = model.train(data=data_yaml, **train_kwargs)
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None) if trainer else None
            if save_dir is None and hasattr(results, "save_dir"):
                save_dir = getattr(results, "save_dir")
            result_holder["save_dir"] = save_dir
            result_holder["results"] = results
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            writer.flush()
            log_queue.put(LOG_DONE)


@st.cache_data(show_spinner=False)
def collect_run_dirs(root_str: str):
    root = Path(root_str) if root_str else None
    if not root or not root.exists():
        return []
    run_dirs = []
    for result_csv in root.rglob("results.csv"):
        run_dirs.append(result_csv.parent)
    unique = sorted({p.resolve() for p in run_dirs}, key=lambda p: p.stat().st_mtime, reverse=True)
    return unique


def render_run_visualization(run_dir: Path):
    if not run_dir or not run_dir.exists():
        st.info("未找到训练结果目录。")
        return
    st.write(f"结果目录：`{run_dir}`")
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            safe_dataframe(df.tail(20), width='stretch')
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if "epoch" in df.columns:
                df_plot = df.set_index("epoch")[numeric_cols]
            else:
                df_plot = df[numeric_cols]
            if numeric_cols:
                st.line_chart(df_plot)
        except Exception as exc:
            st.warning(f"读取 results.csv 失败：{exc}")

    image_names = [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
    ]
    images = []
    captions = []
    for name in image_names:
        path = run_dir / name
        if path.exists():
            images.append(str(path))
            captions.append(name)
    if images:
        st.image(images, caption=captions, width='stretch')

    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"
        if best.exists() or last.exists():
            st.markdown("**权重文件**")
        if best.exists():
            st.download_button(
                "下载 best.pt",
                data=best.read_bytes(),
                file_name=best.name,
                mime="application/octet-stream",
            )
        if last.exists():
            st.download_button(
                "下载 last.pt",
                data=last.read_bytes(),
                file_name=last.name,
                mime="application/octet-stream",
            )


def render_training_platform():
    st.markdown("<div class='hero-title'>YOLO 可视化训练平台</div>", unsafe_allow_html=True)
    st.caption("选择数据集、设置训练参数、输出可视化结果。")

    try:
        from streamlit_tree_select import tree_select  # type: ignore
    except Exception:
        tree_select = None

    missing = check_train_dependencies()
    if missing:
        st.warning(f"训练依赖未安装：{', '.join(missing)}。请先安装相关库。")

    if "train_name" not in st.session_state:
        st.session_state.train_name = datetime.now().strftime("train_%Y%m%d_%H%M%S")
    if "train_project" not in st.session_state:
        st.session_state.train_project = str(Path.cwd() / "runs" / "train_platform")
    if "train_logs" not in st.session_state:
        st.session_state.train_logs = ""
    if "train_last_run" not in st.session_state:
        st.session_state.train_last_run = ""
    if "train_log_lines" not in st.session_state:
        st.session_state.train_log_lines = []
    if "train_log_file" not in st.session_state:
        st.session_state.train_log_file = ""

    templates_dir = Path.cwd() / "runs" / "train_platform" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_files = sorted(templates_dir.glob("*.json"))
    logs_dir = Path.cwd() / "runs" / "train_platform" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>训练配置</div>", unsafe_allow_html=True)

        st.markdown("**参数模板**")
        template_name = st.text_input("模板名称", value="", key="train_template_name")
        template_labels = [p.stem for p in template_files]
        template_pick = st.selectbox(
            "已有模板",
            options=["(无)"] + template_labels,
            index=0,
            key="train_template_pick",
        )
        template_cols = st.columns(3)
        with template_cols[0]:
            save_template = st.button("保存", width='stretch', key="train_template_save")
        with template_cols[1]:
            load_template = st.button("加载", width='stretch', key="train_template_load")
        with template_cols[2]:
            delete_template = st.button("删除", width='stretch', key="train_template_delete")

        if save_template:
            name = template_name.strip() or datetime.now().strftime("template_%Y%m%d_%H%M%S")
            target = templates_dir / f"{name}.json"
            payload = build_train_template_payload()
            save_template_file(target, payload)
            st.success(f"已保存模板：{name}")

        if load_template and template_pick != "(无)":
            target = templates_dir / f"{template_pick}.json"
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
                st.session_state["train_dataset_root"] = payload.get("dataset_root", st.session_state.get("train_dataset_root"))
                st.session_state["train_dataset_manual"] = payload.get("data_yaml", "")
                st.session_state["train_model_path"] = payload.get("model_path", st.session_state.get("train_model_path"))
                st.session_state["train_project_input"] = payload.get("project", st.session_state.get("train_project_input"))
                st.session_state["train_name_input"] = payload.get("name", st.session_state.get("train_name_input"))
                st.session_state["train_exist_ok"] = payload.get("exist_ok", st.session_state.get("train_exist_ok"))
                st.session_state["train_epochs"] = payload.get("epochs", st.session_state.get("train_epochs"))
                st.session_state["train_imgsz"] = payload.get("imgsz", st.session_state.get("train_imgsz"))
                st.session_state["train_batch"] = payload.get("batch", st.session_state.get("train_batch"))
                st.session_state["train_workers"] = payload.get("workers", st.session_state.get("train_workers"))
                st.session_state["train_device"] = payload.get("device", st.session_state.get("train_device"))
                st.session_state["train_amp"] = payload.get("amp", st.session_state.get("train_amp"))
                st.session_state["train_cache"] = payload.get("cache", st.session_state.get("train_cache"))
                st.session_state["train_resume"] = payload.get("resume", st.session_state.get("train_resume"))
                st.session_state["train_optimizer"] = payload.get("optimizer", st.session_state.get("train_optimizer"))
                st.session_state["train_seed"] = payload.get("seed", st.session_state.get("train_seed"))
                st.session_state["train_patience"] = payload.get("patience", st.session_state.get("train_patience"))
                st.session_state["train_cos_lr"] = payload.get("cos_lr", st.session_state.get("train_cos_lr"))
                st.session_state["train_close_mosaic"] = payload.get("close_mosaic", st.session_state.get("train_close_mosaic"))
                st.session_state["train_save_period"] = payload.get("save_period", st.session_state.get("train_save_period"))
                st.session_state["train_advanced"] = payload.get("advanced_text", st.session_state.get("train_advanced"))
                st.session_state["train_cuda_visible"] = payload.get("cuda_visible_devices", st.session_state.get("train_cuda_visible"))
                st.session_state["train_scan_yaml"] = payload.get("scan_yaml", st.session_state.get("train_scan_yaml"))
                st.success(f"已加载模板：{template_pick}")
                trigger_rerun()
            except Exception as exc:
                st.error(f"加载模板失败：{exc}")

        if delete_template and template_pick != "(无)":
            target = templates_dir / f"{template_pick}.json"
            try:
                target.unlink(missing_ok=True)
                st.success(f"已删除模板：{template_pick}")
                trigger_rerun()
            except Exception as exc:
                st.error(f"删除模板失败：{exc}")

        st.markdown("---")

        dataset_default = Path.cwd() / "runs" / "latest" / "yolo_datasets"
        dataset_root_input = st.text_input(
            "训练数据集文件夹",
            value=str(dataset_default) if dataset_default.exists() else str(Path.cwd()),
            key="train_dataset_root",
        )
        dataset_root_suggestions = get_path_suggestions(dataset_root_input, include_hidden=False)
        if dataset_root_suggestions:
            dataset_root = st.selectbox(
                "路径自动补全",
                options=dataset_root_suggestions,
                index=0,
                key="train_dataset_root_suggest",
            )
            if dataset_root != dataset_root_input:
                st.session_state["train_dataset_root"] = dataset_root
        else:
            dataset_root = dataset_root_input
        if not Path(dataset_root).exists():
            st.warning("当前路径不存在，请检查或使用下方浏览器选择。")
        with st.expander("浏览目录", expanded=False):
            browse_root_input = st.text_input(
                "浏览起点",
                value=str(Path.cwd()),
                key="train_browse_root",
            )
            browse_suggestions = get_path_suggestions(browse_root_input, include_hidden=False)
            if browse_suggestions:
                browse_root = st.selectbox(
                    "自动补全",
                    options=browse_suggestions,
                    index=0,
                    key="train_browse_root_suggest",
                )
                if browse_root != browse_root_input:
                    st.session_state["train_browse_root"] = browse_root
            else:
                browse_root = browse_root_input
            show_hidden = st.checkbox("显示隐藏目录", value=False, key="train_browse_hidden")
            st.caption("推荐：使用树状选择快速定位目录。")
            use_tree = st.checkbox("启用树状目录选择", value=True, key="train_browse_use_tree")
            use_file_manager = st.checkbox("文件管理器风格（搜索/面包屑/最近路径）", value=True, key="train_use_file_manager")
            if use_file_manager:
                current_path = Path(st.session_state.get("train_dataset_root") or browse_root)
                if not current_path.exists():
                    current_path = Path(browse_root)
                st.markdown("<div class='file-manager'>", unsafe_allow_html=True)
                shortcut_enabled = st.checkbox("启用快捷键", value=True, key="train_shortcuts_enable")
                st.caption("快捷键：Alt+E 展开 | Alt+W 收起 | Alt+F 收藏 | Alt+S 设为根 | Alt+P 预览 | Alt+U 上一级 | Alt+C 复制路径")
                if shortcut_enabled:
                    key_event = components.html(
                        """
                        <script>
                        document.addEventListener('keydown', function(e) {
                          if (!e.altKey || e.repeat) return;
                          const key = e.key;
                          let value = '';
                          if (key === 'ArrowUp') value = 'alt+u';
                          else value = 'alt+' + key.toLowerCase();
                          window.parent.postMessage({
                            isStreamlitMessage: true,
                            type: 'streamlit:setComponentValue',
                            value
                          }, '*');
                          e.preventDefault();
                        }, { once: false });
                        </script>
                        """,
                        height=0,
                    )
                    if isinstance(key_event, str) and key_event:
                        needs_rerun = False
                        selected_path = st.session_state.get("train_preview_path") or str(current_path)
                        active_group = st.session_state.get("train_fav_group", "默认")
                        if key_event == "alt+e":
                            all_paths = collect_dir_paths(current_path, show_hidden, st.session_state.get("train_icon_tree_depth", 3), st.session_state.get("train_icon_tree_nodes", 300))
                            st.session_state["train_icon_tree_expanded"] = all_paths
                            needs_rerun = True
                        elif key_event == "alt+w":
                            st.session_state["train_icon_tree_expanded"] = []
                            needs_rerun = True
                        elif key_event == "alt+f":
                            add_favorite_path(selected_path, group=active_group)
                            needs_rerun = True
                        elif key_event == "alt+s":
                            st.session_state["train_dataset_root"] = selected_path
                            add_recent_path(selected_path)
                            needs_rerun = True
                        elif key_event == "alt+p":
                            st.session_state["train_preview_path"] = selected_path
                            needs_rerun = True
                        elif key_event == "alt+u":
                            parent = str(Path(selected_path).parent)
                            st.session_state["train_dataset_root"] = parent
                            st.session_state["train_preview_path"] = parent
                            add_recent_path(parent)
                            needs_rerun = True
                        elif key_event == "alt+c":
                            copy_to_clipboard(selected_path)
                        if needs_rerun:
                            trigger_rerun()
                left_col, right_col = st.columns([1.1, 1.4])
                with left_col:
                    st.caption("当前位置")
                    st.markdown(f"<div class='fm-path'>{current_path}</div>", unsafe_allow_html=True)
                    crumbs = []
                    accum = Path(current_path.anchor) if current_path.anchor else Path("/")
                    for part in current_path.parts:
                        if part == current_path.anchor:
                            crumbs.append((part, str(accum)))
                            continue
                        accum = accum / part
                        crumbs.append((part, str(accum)))
                    if crumbs:
                        st.caption("面包屑")
                        crumb_cols = st.columns(min(len(crumbs), 6))
                        for idx, (label, path_value) in enumerate(crumbs[:6]):
                            with crumb_cols[idx]:
                                if st.button(label if label else "/", key=f"train_crumb_{idx}", width='stretch'):
                                    st.session_state["train_dataset_root"] = path_value
                                    add_recent_path(path_value)
                                    trigger_rerun()
                    groups = ensure_favorite_groups()
                    group_names = list(groups.keys())
                    if not group_names:
                        group_names = ["默认"]
                    active_group = st.selectbox("收藏组", options=group_names, index=0, key="train_fav_group")
                    new_group = st.text_input("新建收藏组", value="", key="train_fav_new_group")
                    group_cols = st.columns(2)
                    with group_cols[0]:
                        if st.button("添加组", key="train_fav_add_group", width='stretch'):
                            add_favorite_group(new_group.strip())
                            trigger_rerun()
                    with group_cols[1]:
                        if st.button("删除组", key="train_fav_del_group", width='stretch'):
                            delete_favorite_group(active_group)
                            trigger_rerun()

                    if groups:
                        st.caption("收藏路径（分组）")
                        for group_name, paths in groups.items():
                            with st.expander(f"{group_name} ({len(paths)})", expanded=False):
                                for idx, path_value in enumerate(paths):
                                    fav_cols = st.columns([3.2, 1])
                                    with fav_cols[0]:
                                        if st.button(path_value, key=f"train_fav_{group_name}_{idx}", width='stretch'):
                                            st.session_state["train_dataset_root"] = path_value
                                            trigger_rerun()
                                    with fav_cols[1]:
                                        if st.button("移除", key=f"train_fav_remove_{group_name}_{idx}", width='stretch'):
                                            remove_favorite_path(path_value, group_name)
                                            trigger_rerun()

                    recent = st.session_state.get("train_recent_paths", [])
                    if recent:
                        st.caption("最近路径")
                        for idx, path_value in enumerate(recent):
                            if st.button(path_value, key=f"train_recent_{idx}", width='stretch'):
                                st.session_state["train_dataset_root"] = path_value
                                trigger_rerun()

                    search_query = st.text_input("搜索目录", value="", key="train_dir_search")
                    if search_query.strip():
                        results = search_directories(current_path, search_query.strip(), show_hidden, max_results=40)
                        if results:
                            st.caption(f"找到 {len(results)} 个目录")
                            for idx, path_value in enumerate(results):
                                if st.button(str(path_value), key=f"train_search_{idx}", width='stretch'):
                                    st.session_state["train_dataset_root"] = str(path_value)
                                    add_recent_path(str(path_value))
                                    trigger_rerun()
                        else:
                            st.info("未找到匹配目录。")

                    st.caption("目录树模式")
                    tree_mode = st.radio(
                        "选择目录树",
                        options=["图标树(拖拽)", "高级树(右键菜单)"],
                        index=0,
                        key="train_tree_mode",
                        horizontal=True,
                    )
                    tree_filter = st.text_input("目录过滤", value="", key="train_tree_filter")
                    icon_tree_depth = st.slider("图标树深度", min_value=1, max_value=6, value=3, step=1, key="train_icon_tree_depth")
                    icon_tree_nodes = st.slider("图标树节点上限", min_value=50, max_value=1200, value=300, step=50, key="train_icon_tree_nodes")
                    tree_action_cols = st.columns(3)
                    with tree_action_cols[0]:
                        if st.button("展开全部", key="train_icon_expand_all", width='stretch'):
                            all_paths = collect_dir_paths(current_path, show_hidden, icon_tree_depth, icon_tree_nodes)
                            st.session_state["train_icon_tree_expanded"] = all_paths
                            trigger_rerun()
                    with tree_action_cols[1]:
                        if st.button("收起全部", key="train_icon_collapse_all", width='stretch'):
                            st.session_state["train_icon_tree_expanded"] = []
                            trigger_rerun()
                    with tree_action_cols[2]:
                        if st.button("清除选中", key="train_icon_clear_sel", width='stretch'):
                            st.session_state["train_icon_tree_selected"] = ""
                            trigger_rerun()

                    if tree_mode == "高级树(右键菜单)":
                        nodes, root_id = build_tree_flat(current_path, show_hidden, icon_tree_depth, icon_tree_nodes)
                        nodes = filter_tree_nodes(nodes, tree_filter.strip())
                        adv_value = render_advanced_tree_component(
                            nodes,
                            root_id,
                            st.session_state.get("train_adv_tree_expanded", []),
                            st.session_state.get("train_adv_tree_selected", ""),
                        )
                        if isinstance(adv_value, str) and adv_value:
                            try:
                                payload = json.loads(adv_value)
                            except Exception:
                                payload = {}
                            action = payload.get("action")
                            path = payload.get("path")
                            expanded = payload.get("expanded")
                            selected = payload.get("selected")
                            if expanded is not None:
                                st.session_state["train_adv_tree_expanded"] = expanded
                            if selected:
                                st.session_state["train_adv_tree_selected"] = selected
                            if action in {"select", "preview"} and path:
                                st.session_state["train_preview_path"] = path
                                add_recent_path(path)
                                trigger_rerun()
                            if action == "set_root" and path:
                                st.session_state["train_dataset_root"] = path
                                st.session_state["train_browse_root"] = path
                                add_recent_path(path)
                                trigger_rerun()
                    else:
                        render_icon_tree_custom(current_path, show_hidden, icon_tree_depth, icon_tree_nodes, filter_query=tree_filter)
                    st.markdown("---")
                    if st.button("收藏当前路径", key="train_fav_add", width='stretch'):
                        add_favorite_path(str(current_path), group=active_group)
                        trigger_rerun()
                with right_col:
                    st.caption("目录信息")
                    selected_path = st.session_state.get("train_preview_path") or str(current_path)
                    selected_dir = Path(selected_path)
                    st.markdown(f"<div class='fm-path'>{selected_dir}</div>", unsafe_allow_html=True)
                    if selected_dir.exists():
                        subdir_count = len([p for p in selected_dir.iterdir() if p.is_dir()])
                        file_count = len([p for p in selected_dir.iterdir() if p.is_file()])
                        st.write(f"子目录：{subdir_count}")
                        st.write(f"文件：{file_count}")
                        stats_recursive = st.checkbox("递归统计大小", value=False, key="train_stats_recursive")
                        stats_depth = st.slider("统计深度", min_value=1, max_value=10, value=6, step=1, key="train_stats_depth")
                        stats_limit = st.number_input("统计文件上限", min_value=500, max_value=50000, value=5000, step=500, key="train_stats_limit")
                        stats = get_dir_stats(
                            selected_dir,
                            recursive=stats_recursive,
                            max_files=int(stats_limit),
                            max_depth=int(stats_depth),
                        )
                        st.write(f"大小：{format_bytes(stats.get('bytes'))}")
                        st.write(f"统计文件数：{stats.get('files')} · 目录数：{stats.get('dirs')}")
                        if stats.get("truncated"):
                            st.caption("已达到统计上限，结果为近似值。")
                        st.caption("大小占比（非递归，按当前层）")
                        dir_sizes, file_sizes = get_immediate_children_sizes(selected_dir, max_items=6)
                        if dir_sizes:
                            max_dir = max(size for _, size in dir_sizes) or 1
                            for name, size in dir_sizes:
                                st.write(f"{name} · {format_bytes(size)}")
                                st.progress(min(size / max_dir, 1.0))
                        if file_sizes:
                            max_file = max(size for _, size in file_sizes) or 1
                            for name, size in file_sizes:
                                st.write(f"{name} · {format_bytes(size)}")
                                st.progress(min(size / max_file, 1.0))
                    st.caption("预览选中目录缩略图")
                    drop_target = components.html(
                        """
                        <div class="drop-zone" id="drop-zone">拖拽目录到这里预览</div>
                        <script>
                        const dz = document.getElementById('drop-zone');
                        dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.style.background = 'rgba(59,130,246,0.12)'; });
                        dz.addEventListener('dragleave', (e) => { dz.style.background = 'rgba(59,130,246,0.05)'; });
                        dz.addEventListener('drop', (e) => {
                          e.preventDefault();
                          dz.style.background = 'rgba(59,130,246,0.05)';
                          const path = e.dataTransfer.getData('text/plain');
                          window.parent.postMessage({
                            isStreamlitMessage: true,
                            type: 'streamlit:setComponentValue',
                            value: path
                          }, '*');
                        });
                        </script>
                        """,
                        height=70,
                    )
                    if isinstance(drop_target, str) and drop_target:
                        st.session_state["train_preview_path"] = drop_target
                        if st.checkbox("拖拽同时更新根目录", value=False, key="train_drop_set_root"):
                            st.session_state["train_dataset_root"] = drop_target
                            add_recent_path(drop_target)
                        trigger_rerun()

                    preview_recursive = st.checkbox("递归搜索图片", value=False, key="train_preview_recursive")
                    preview_scan_limit = st.number_input("扫描上限", min_value=50, max_value=10000, value=600, step=50, key="train_preview_scan_limit")
                    preview_page_size = st.slider("每页数量", min_value=4, max_value=48, value=12, step=4, key="train_preview_page_size")
                    st.caption(f"预览目录：`{selected_dir}`")
                    image_files = list_image_files_for_preview(str(selected_dir), preview_recursive, int(preview_scan_limit))
                    search_name = st.text_input("文件名搜索", value="", key="train_preview_search")
                    sort_by = st.selectbox("排序方式", options=["名称", "修改时间", "大小"], index=0, key="train_preview_sort")
                    sort_order = st.selectbox("顺序", options=["升序", "降序"], index=0, key="train_preview_order")
                    filtered = image_files
                    if search_name.strip():
                        key = search_name.strip().lower()
                        filtered = [item for item in filtered if key in Path(item["path"]).name.lower()]
                    reverse = sort_order == "降序"
                    if sort_by == "名称":
                        filtered = sorted(filtered, key=lambda x: Path(x["path"]).name.lower(), reverse=reverse)
                    elif sort_by == "修改时间":
                        filtered = sorted(filtered, key=lambda x: x.get("mtime", 0), reverse=reverse)
                    else:
                        filtered = sorted(filtered, key=lambda x: x.get("size", 0), reverse=reverse)
                    total_images = len(filtered)
                    total_pages = max(1, math.ceil(total_images / preview_page_size)) if total_images else 1
                    page_cols = st.columns([1, 1, 2])
                    with page_cols[0]:
                        if st.button("上一页", key="train_preview_prev", width='stretch'):
                            current_page = max(1, st.session_state.get("train_preview_page", 1) - 1)
                            st.session_state["train_preview_page"] = current_page
                            trigger_rerun()
                    with page_cols[1]:
                        if st.button("下一页", key="train_preview_next", width='stretch'):
                            current_page = min(total_pages, st.session_state.get("train_preview_page", 1) + 1)
                            st.session_state["train_preview_page"] = current_page
                            trigger_rerun()
                    with page_cols[2]:
                        current_page = st.number_input("页码", min_value=1, max_value=total_pages, value=int(st.session_state.get("train_preview_page", 1)), step=1, key="train_preview_page_input")
                        st.session_state["train_preview_page"] = current_page

                    lazy_scroll = st.checkbox("滚动加载下一页", value=False, key="train_preview_lazy_scroll")
                    if lazy_scroll:
                        scroll_event = components.html(
                            """
                            <script>
                            let ticking = false;
                            function onScroll() {
                              if (ticking) return;
                              ticking = true;
                              window.requestAnimationFrame(() => {
                                const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
                                const scrollHeight = document.documentElement.scrollHeight || document.body.scrollHeight;
                                const clientHeight = document.documentElement.clientHeight || document.body.clientHeight;
                                if (scrollTop + clientHeight >= scrollHeight - 120) {
                                  window.parent.postMessage({
                                    isStreamlitMessage: true,
                                    type: 'streamlit:setComponentValue',
                                    value: 'next'
                                  }, '*');
                                }
                                ticking = false;
                              });
                            }
                            window.addEventListener('scroll', onScroll, { passive: true });
                            </script>
                            """,
                            height=0,
                        )
                        if isinstance(scroll_event, str) and scroll_event == "next":
                            if current_page < total_pages:
                                st.session_state["train_preview_page"] = current_page + 1
                                trigger_rerun()

                    start = (current_page - 1) * preview_page_size
                    end = start + preview_page_size
                    page_files = filtered[start:end]
                    if total_images == 0:
                        st.info("未找到可预览图片。")
                    else:
                        st.caption(f"已扫描 {total_images} 张，显示 {start + 1}-{min(end, total_images)}")
                        view_mode = st.radio("预览模式", options=["网格", "列表"], index=0, key="train_preview_mode", horizontal=True)
                        if view_mode == "网格":
                            grid_mode_type = st.selectbox("列数模式", options=["自适应", "手动"], index=0, key="train_preview_grid_mode")
                            if grid_mode_type == "自适应":
                                cols_count = min(6, max(2, int(math.sqrt(preview_page_size))))
                            else:
                                cols_count = st.slider("列数", min_value=2, max_value=8, value=3, step=1, key="train_preview_grid_cols")
                            cols = st.columns(cols_count)
                            for idx, item in enumerate(page_files):
                                with cols[idx % cols_count]:
                                    st.image(item["path"], caption=Path(item["path"]).name, width='stretch')
                        else:
                            rows = []
                            for item in page_files:
                                rows.append({
                                    "文件名": Path(item["path"]).name,
                                    "大小": format_bytes(item.get("size", 0)),
                                    "修改时间": datetime.fromtimestamp(item.get("mtime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                                    "路径": item["path"],
                                })
                            safe_dataframe(pd.DataFrame(rows), width='stretch')
                        if total_images >= int(preview_scan_limit) and st.button("加载更多", key="train_preview_load_more", width='stretch'):
                            st.session_state["train_preview_scan_limit"] = int(preview_scan_limit) + 200
                            list_image_files_for_preview.clear()
                            trigger_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            if use_tree and tree_select:
                tree_depth = st.slider("树状深度", min_value=1, max_value=8, value=4, step=1, key="train_tree_depth")
                tree_nodes_limit = st.slider("最大节点数", min_value=200, max_value=5000, value=2000, step=200, key="train_tree_nodes")
                if st.button("刷新树", key="train_tree_refresh", width='stretch'):
                    build_dir_tree_nodes_cached.clear()
                    trigger_rerun()
                nodes, total_nodes = build_dir_tree_nodes_cached(browse_root, tree_depth, tree_nodes_limit, show_hidden)
                if not nodes:
                    st.caption("树状目录为空，请检查浏览起点。")
                else:
                    st.caption(f"已加载 {total_nodes} 个目录节点")
                    tree_state = tree_select(
                        nodes,
                        check_model="all",
                        only_leaf_checkboxes=False,
                        no_cascade=True,
                        expand_on_click=True,
                        show_expand_all=True,
                        expanded=st.session_state.get("train_tree_expanded", []),
                        checked=st.session_state.get("train_tree_checked", []),
                    )
                    checked = tree_state.get("checked", []) if isinstance(tree_state, dict) else []
                    expanded = tree_state.get("expanded", []) if isinstance(tree_state, dict) else []
                    st.session_state["train_tree_expanded"] = expanded
                    if checked:
                        st.session_state["train_tree_checked"] = [checked[-1]]
                        selected_path = checked[-1]
                        st.caption(f"树状选择：`{selected_path}`")
                        if st.button("使用树状路径", key="train_use_tree_path", width='stretch'):
                            st.session_state["train_dataset_root"] = selected_path
                            add_recent_path(selected_path)
                            trigger_rerun()
            elif use_tree and not tree_select:
                st.info("树状目录需要安装 streamlit-tree-select。")
            base_path = Path(browse_root)
            if not base_path.exists() or not base_path.is_dir():
                st.caption("浏览起点不存在或不是目录。")
            else:
                if st.session_state.get("train_browse_base_cache") != str(base_path):
                    st.session_state["train_browse_base_cache"] = str(base_path)
                    st.session_state["train_browse_stack"] = []
                stack = list(st.session_state.get("train_browse_stack", []))
                max_depth = st.slider("目录层级深度", min_value=1, max_value=8, value=4, step=1, key="train_browse_depth")
                current_path = str(base_path)
                for level in range(max_depth):
                    parent = Path(current_path)
                    subdirs = list_subdirectories(str(parent), include_hidden=show_hidden)
                    if not subdirs:
                        break
                    options = ["(当前)"] + [str(p) for p in subdirs]
                    current_value = stack[level] if level < len(stack) else "(当前)"
                    index = options.index(current_value) if current_value in options else 0
                    choice = st.selectbox(
                        f"第 {level + 1} 级",
                        options=options,
                        index=index,
                        key=f"train_browse_level_{level}",
                    )
                    if choice == "(当前)":
                        stack = stack[:level]
                        break
                    if len(stack) <= level:
                        stack.append(choice)
                    else:
                        stack[level] = choice
                        stack = stack[: level + 1]
                    current_path = choice
                st.session_state["train_browse_stack"] = stack
                selected_path = stack[-1] if stack else str(base_path)
                st.caption(f"当前选择：`{selected_path}`")
                browse_cols = st.columns(3)
                with browse_cols[0]:
                    if st.button("使用当前路径", key="train_use_browse", width='stretch'):
                        st.session_state["train_dataset_root"] = selected_path
                        trigger_rerun()
                with browse_cols[1]:
                    if st.button("上一级", key="train_browse_up", width='stretch'):
                        if stack:
                            stack = stack[:-1]
                            st.session_state["train_browse_stack"] = stack
                        trigger_rerun()
                with browse_cols[2]:
                    if st.button("重置", key="train_browse_reset", width='stretch'):
                        st.session_state["train_browse_stack"] = []
                        trigger_rerun()
        scan_yaml = st.checkbox("扫描 data.yaml / dataset.yaml", value=True, key="train_scan_yaml")
        if st.button("重新扫描", key="train_rescan", width='stretch'):
            scan_dataset_configs.clear()
            trigger_rerun()
        dataset_root_path = Path(dataset_root)
        if dataset_root_path.is_file() and dataset_root_path.suffix.lower() in {".yaml", ".yml"}:
            dataset_yaml_options = [dataset_root_path]
        else:
            dataset_yaml_options = scan_dataset_configs(dataset_root) if scan_yaml else []
        dataset_yaml_choice = None
        if dataset_yaml_options:
            dataset_yaml_choice = st.selectbox(
                "选择数据集配置文件",
                options=[str(p) for p in dataset_yaml_options],
                key="train_dataset_choice",
            )
        else:
            st.caption("未找到 data.yaml / dataset.yaml，请检查目录或手动输入路径。")
        manual_yaml = st.text_input("或手动输入 data.yaml 路径", value="", key="train_dataset_manual")
        if manual_yaml.strip():
            st.info("已填写手动路径，将优先使用。清空后才能使用下拉选择。")
            if st.button("清空手动路径", key="train_clear_manual", width='stretch'):
                st.session_state["train_dataset_manual"] = ""
                trigger_rerun()
        with st.expander("浏览 data.yaml", expanded=False):
            yaml_files = list_yaml_files(dataset_root)
            if yaml_files:
                pick_yaml = st.selectbox(
                    "选择 data.yaml 文件",
                    options=[str(p) for p in yaml_files],
                    key="train_browse_yaml",
                )
                if st.button("使用选中文件", key="train_use_yaml", width='stretch'):
                    st.session_state["train_dataset_manual"] = pick_yaml
                    trigger_rerun()
            else:
                st.caption("当前目录未找到 data.yaml / dataset.yaml。")
        data_yaml = manual_yaml.strip() or dataset_yaml_choice or ""

        st.markdown("---")
        st.markdown("<div class='sidebar-title'>模型与输出</div>", unsafe_allow_html=True)
        model_path = st.text_input(
            "模型/权重路径",
            value="ultralytics/cfg/models/11/yolo11.yaml",
            key="train_model_path",
        )
        project = st.text_input(
            "输出目录 project",
            value=st.session_state.train_project,
            key="train_project_input",
        )
        name = st.text_input(
            "训练名称 name",
            value=st.session_state.train_name,
            key="train_name_input",
        )
        exist_ok = st.checkbox("exist_ok（覆盖同名结果）", value=False, key="train_exist_ok")

        st.markdown("---")
        st.markdown("<div class='sidebar-title'>基础参数</div>", unsafe_allow_html=True)
        epochs = st.number_input("epochs", min_value=1, max_value=5000, value=50, step=1, key="train_epochs")
        imgsz = st.number_input("imgsz", min_value=320, max_value=4096, value=640, step=32, key="train_imgsz")
        batch = st.number_input("batch", min_value=1, max_value=1024, value=16, step=1, key="train_batch")
        workers = st.number_input("workers", min_value=0, max_value=64, value=4, step=1, key="train_workers")
        device = st.text_input("device（如 0 / 0,1 / cpu）", value="0", key="train_device")
        amp = st.checkbox("AMP 混合精度", value=True, key="train_amp")
        cache_choice = st.selectbox("cache", options=["False", "True", "ram", "disk"], index=0, key="train_cache")
        resume = st.checkbox("resume（断点续训）", value=False, key="train_resume")

        st.markdown("---")
        st.markdown("<div class='sidebar-title'>进阶参数</div>", unsafe_allow_html=True)
        optimizer = st.text_input("optimizer", value="auto", key="train_optimizer")
        seed = st.number_input("seed", min_value=0, max_value=999999, value=0, step=1, key="train_seed")
        patience = st.number_input("patience", min_value=0, max_value=500, value=50, step=1, key="train_patience")
        cos_lr = st.checkbox("cos_lr", value=False, key="train_cos_lr")
        close_mosaic = st.number_input("close_mosaic", min_value=0, max_value=200, value=10, step=1, key="train_close_mosaic")
        save_period = st.number_input("save_period", min_value=-1, max_value=200, value=-1, step=1, key="train_save_period")

        st.markdown("---")
        st.markdown("<div class='sidebar-title'>高级参数</div>", unsafe_allow_html=True)
        advanced_text = st.text_area(
            "key=value（支持数字/true/false/[]/{})",
            value="",
            height=140,
            key="train_advanced",
        )
        cuda_visible_devices = st.text_input(
            "CUDA_VISIBLE_DEVICES（可选）",
            value="",
            key="train_cuda_visible",
        )

        st.markdown("---")
        st.markdown("<div class='sidebar-title'>训练监控</div>", unsafe_allow_html=True)
        stream_logs = st.checkbox("实时日志流", value=True, key="train_stream_logs")
        max_log_lines = st.number_input("日志保留行数", min_value=200, max_value=5000, value=1200, step=100, key="train_log_lines_limit")

    dataset_summary = summarize_dataset(data_yaml) if data_yaml else {"error": "未选择数据集配置"}
    cuda_info = get_cuda_summary()

    info_left, info_right = st.columns([2, 1])
    with info_left:
        st.markdown("**数据集概览**")
        if dataset_summary.get("error"):
            st.warning(dataset_summary["error"])
        else:
            st.write(f"data.yaml：`{data_yaml}`")
            st.write(f"数据集根目录：`{dataset_summary.get('path')}`")
            st.write(f"类别数：{dataset_summary.get('nc')}")
            names = dataset_summary.get("names")
            if names:
                st.write(f"类别：{', '.join([str(n) for n in names])}")
            st.write(f"训练集图片：{format_int_safe(dataset_summary.get('train_images'))}")
            st.write(f"验证集图片：{format_int_safe(dataset_summary.get('val_images'))}")
            st.write(f"测试集图片：{format_int_safe(dataset_summary.get('test_images'))}")

    with info_right:
        st.markdown("**算力信息**")
        if cuda_info.get("available"):
            st.write(f"CUDA 可用：是（{cuda_info.get('detail')}）")
            for idx, name in enumerate(cuda_info.get("devices", [])):
                st.write(f"GPU {idx}: {name}")
        else:
            st.write(f"CUDA 可用：否（{cuda_info.get('detail')}）")

    st.markdown("---")
    with st.expander("数据集可视化浏览器", expanded=False):
        if dataset_summary.get("error"):
            st.info("请先选择有效的 data.yaml 后再浏览。")
        else:
            preview_yaml = data_yaml
            category_options = build_category_preview_options(Path(dataset_root), dataset_yaml_options)
            if category_options:
                category_choice = st.selectbox(
                    "类别过滤子目录",
                    options=["(当前数据集)"] + list(category_options.keys()),
                    index=0,
                    key="train_preview_category",
                )
                if category_choice != "(当前数据集)":
                    preview_yaml = category_options.get(category_choice, preview_yaml)

            preview_summary = summarize_dataset(preview_yaml) if preview_yaml else {"error": "未选择 data.yaml"}
            if preview_summary.get("error"):
                st.warning(preview_summary["error"])
            else:
                st.caption(f"使用 data.yaml：`{preview_yaml}`")
                split_choice = st.selectbox("选择 split", options=["train", "val", "test"], index=0, key="train_preview_split")
                max_preview = st.slider("缩略图数量", min_value=4, max_value=64, value=16, step=4, key="train_preview_count")
                shuffle_preview = st.checkbox("随机抽样", value=True, key="train_preview_shuffle")
                dir_map = {
                    "train": Path(preview_summary.get("train_dir", "")),
                    "val": Path(preview_summary.get("val_dir", "")),
                    "test": Path(preview_summary.get("test_dir", "")),
                }
                target_dir = dir_map.get(split_choice)
                st.caption(f"目录：`{target_dir}`")
                images = collect_image_files(target_dir, max_images=int(max_preview), shuffle=shuffle_preview)
                if images:
                    st.image([str(p) for p in images], caption=[p.name for p in images], width='stretch')
                else:
                    st.info("该 split 未找到可预览图片。")

    st.markdown("---")
    train_btn = st.button("开始训练", type="primary", width='stretch', disabled=bool(missing))

    if train_btn:
        if not data_yaml:
            st.error("请先选择或输入 data.yaml 路径。")
            st.stop()
        if not Path(data_yaml).exists():
            st.error("data.yaml 不存在，请检查路径。")
            st.stop()

        cache_value = {"False": False, "True": True}.get(cache_choice, cache_choice)
        train_kwargs = {
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": int(workers),
            "device": device.strip() or None,
            "project": project.strip() or None,
            "name": name.strip() or None,
            "exist_ok": bool(exist_ok),
            "amp": bool(amp),
            "cache": cache_value,
            "resume": bool(resume),
            "optimizer": optimizer.strip() or "auto",
            "seed": int(seed),
            "patience": int(patience),
            "cos_lr": bool(cos_lr),
            "close_mosaic": int(close_mosaic),
            "save_period": int(save_period),
        }
        if train_kwargs.get("device") is None:
            train_kwargs.pop("device", None)
        if train_kwargs.get("project") is None:
            train_kwargs.pop("project", None)
        if train_kwargs.get("name") is None:
            train_kwargs.pop("name", None)

        advanced_opts, errors = parse_kv_lines(advanced_text)
        if errors:
            st.warning("高级参数解析提示：" + "；".join(errors))
        if advanced_opts:
            train_kwargs.update(advanced_opts)

        env_vars = {}
        if cuda_visible_devices.strip():
            env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()

        progress_bar = st.progress(0.0)
        log_placeholder = st.empty()
        status_placeholder = st.empty()
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{safe_filename(name.strip() if name else 'train')}_{run_stamp}.log"
        log_file_path = logs_dir / log_file_name
        st.session_state.train_log_file = str(log_file_path)

        if stream_logs:
            log_queue = queue.Queue()
            result_holder = {}
            worker = threading.Thread(
                target=run_yolo_training_stream,
                args=(model_path, data_yaml, train_kwargs, env_vars, log_queue, result_holder),
                daemon=True,
            )
            worker.start()

            log_lines = list(st.session_state.train_log_lines or [])
            max_lines = int(max_log_lines)
            total_epochs = int(train_kwargs.get("epochs", 0)) or None
            current_epoch = 0
            log_file = None
            try:
                log_file = open(log_file_path, "a", encoding="utf-8")
            except Exception as exc:
                st.warning(f"无法写入日志文件：{exc}")

            status_placeholder.info("训练进行中（实时日志流已开启）…")
            done = False
            while not done:
                try:
                    item = log_queue.get(timeout=0.2)
                except queue.Empty:
                    item = None

                if item is LOG_DONE:
                    done = True
                elif isinstance(item, str):
                    if item.strip():
                        log_lines.append(item)
                        if len(log_lines) > max_lines:
                            log_lines = log_lines[-max_lines:]
                        st.session_state.train_log_lines = log_lines
                        st.session_state.train_logs = "\n".join(log_lines)
                        if log_file:
                            try:
                                log_file.write(item + "\n")
                                log_file.flush()
                            except Exception:
                                log_file = None
                        epoch_info = _extract_epoch_info(item)
                        if epoch_info:
                            current_epoch, total_epochs = epoch_info
                if total_epochs:
                    progress_bar.progress(min(current_epoch / total_epochs, 1.0))
                log_placeholder.text_area("训练输出（实时）", st.session_state.train_logs, height=260)

                if item is None and not worker.is_alive():
                    done = True
            if log_file:
                log_file.close()

            save_dir = result_holder.get("save_dir")
            error = result_holder.get("error")
            st.session_state.train_last_run = str(save_dir) if save_dir else ""
            if error:
                st.error(f"训练失败：{error}")
            else:
                st.success("训练完成！")
                payload = build_train_template_payload()
                try:
                    save_template_file(templates_dir / "last_success.json", payload)
                except Exception as exc:
                    st.warning(f"写入 last_success 模板失败：{exc}")
                collect_run_dirs.clear()
        else:
            status_placeholder.info("训练进行中（实时日志流已关闭）…")
            with st.spinner("训练中，请耐心等待……"):
                _, logs, save_dir, error = run_yolo_training(model_path, data_yaml, train_kwargs, env_vars)
            try:
                log_file_path.write_text(logs, encoding="utf-8")
            except Exception as exc:
                st.warning(f"无法写入日志文件：{exc}")
            lines = logs.splitlines()
            max_lines = int(max_log_lines)
            st.session_state.train_log_lines = lines[-max_lines:] if len(lines) > max_lines else lines
            st.session_state.train_logs = "\n".join(st.session_state.train_log_lines)
            st.session_state.train_last_run = str(save_dir) if save_dir else ""
            if error:
                st.error(f"训练失败：{error}")
            else:
                st.success("训练完成！")
                payload = build_train_template_payload()
                try:
                    save_template_file(templates_dir / "last_success.json", payload)
                except Exception as exc:
                    st.warning(f"写入 last_success 模板失败：{exc}")
                collect_run_dirs.clear()

    st.markdown("---")
    st.markdown("**训练日志**")
    if st.session_state.train_log_file:
        log_path = Path(st.session_state.train_log_file)
        if log_path.exists():
            st.write(f"日志文件：`{log_path}`")
            st.download_button(
                "下载日志文件",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="text/plain",
            )
    if st.session_state.train_logs:
        st.text_area("训练输出", st.session_state.train_logs, height=260)
    else:
        st.info("暂无日志输出。")

    st.markdown("---")
    st.markdown("**训练结果可视化**")
    run_root = Path(project) if project else Path.cwd() / "runs"
    run_dirs = collect_run_dirs(str(run_root))
    default_run = st.session_state.train_last_run or (str(run_dirs[0]) if run_dirs else "")
    selected_run = None
    if run_dirs:
        index = 0
        if default_run:
            for idx, path in enumerate(run_dirs):
                if str(path) == str(default_run):
                    index = idx
                    break
        selected_run = st.selectbox(
            "选择训练结果目录",
            options=[str(p) for p in run_dirs],
            index=index,
        )
    elif default_run:
        selected_run = default_run
    else:
        st.info("未找到训练结果目录。")

    if selected_run:
        render_run_visualization(Path(selected_run))


inject_style()

BUILD_VERSION = "2026-02-06.1"
st.caption(f"Build: {BUILD_VERSION}")

mode = st.sidebar.radio("功能模式", ["数据处理", "YOLO训练平台"], index=0)
if mode == "YOLO训练平台":
    render_training_platform()
    st.stop()

st.markdown("<div class='hero-title'>YOLO 数据处理流水线</div>", unsafe_allow_html=True)
st.caption("合并CSV → 按source去重 → 参考去重 → 替换ptList → IoU筛选 → 标签替换 → 图片标注")



STEP_ORDER = [
    "merge",
    "dedup",
    "ref_filter",
    "replace_ptlist",
    "iou_filter",
    "label_replace",
    "split",
    "yolo",
    "download",
]


def init_state():
    if "run_id" not in st.session_state:
        st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fixed_output_root = Path.cwd() / "runs" / "latest"
    if "output_root" not in st.session_state:
        st.session_state.output_root = str(fixed_output_root)
    if "outputs" not in st.session_state:
        st.session_state.outputs = {}
    if "logs" not in st.session_state:
        st.session_state.logs = {}
    if "step_done" not in st.session_state:
        st.session_state.step_done = {}
    if "input_ready" not in st.session_state:
        st.session_state.input_ready = False
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "preview_path" not in st.session_state:
        st.session_state.preview_path = None


init_state()
FIXED_OUTPUT_ROOT = Path(st.session_state.output_root)


# def save_upload(uploaded_file, dest_path: Path):
#     dest_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(dest_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return dest_path
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_upload(uploaded_file, dest_path: Path):
    """
    修复版：上传文件保存方法，增加异常捕获、校验和日志
    """
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

def check_requirements():
    req_path = Path(__file__).resolve().parent / "requirements.txt"
    if not req_path.exists():
        return ["requirements.txt 未找到"]
    mapping = {
        "Pillow": "PIL",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
    }
    missing = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("==")[0].strip()
        module = mapping.get(pkg, pkg)
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)
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


missing_pkgs = check_requirements()
if missing_pkgs:
    st.warning(f"环境依赖缺失：{', '.join(missing_pkgs)}。请先安装 requirements.txt。")


def save_uploads(uploaded_files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in uploaded_files:
        out_path = dest_dir / item.name
        save_upload(item, out_path)
        paths.append(out_path)
    return paths


def file_info_from_upload(uploaded_file):
    size = getattr(uploaded_file, "size", None)
    if size is None:
        try:
            size = len(uploaded_file.getbuffer())
        except Exception:
            size = 0
    return {
        "name": uploaded_file.name,
        "size_kb": size / 1024,
        "type": getattr(uploaded_file, "type", "未知类型") or "未知类型",
    }


def file_info_from_path(path: Path):
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    suffix = path.suffix.lower().lstrip(".")
    file_type = suffix if suffix else "文件"
    return {
        "name": path.name,
        "size_kb": size / 1024,
        "type": file_type,
    }


def render_file_tiles(title, file_infos, columns=3):
    if not file_infos:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(columns)
    for idx, info in enumerate(file_infos):
        with cols[idx % columns]:
            st.markdown(
                f"""
                <div class="file-card">
                  <div class="file-name">{info['name']}</div>
                  <div class="file-meta">{info['size_kb']:.1f} KB</div>
                  <div class="file-meta">类型：{info['type']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_step(step_key, step_name, func, *args, **kwargs):
    buffer = io.StringIO()
    busy = st.empty()
    busy.markdown(
        f"<div class='busy-indicator'>正在执行：{step_name} <span class='busy-dots'><span></span><span></span><span></span></span></div>",
        unsafe_allow_html=True,
    )
    with st.spinner(""):
        with redirect_stdout(buffer):
            result = func(*args, **kwargs)
    busy.empty()
    st.session_state.logs[step_key] = buffer.getvalue()
    st.success(f"{step_name} 完成")
    return result


def show_logs(step_key, step_name):
    logs = st.session_state.logs.get(step_key)
    if logs:
        st.text_area(f"{step_name} 日志", logs, height=180)


def preview_csv(path: Path, label: str):
    if path and path.exists():
        st.write(f"{label}：`{path}`")
        try:
            if str(path).lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path, nrows=200)
            else:
                df = pd.read_csv(path, nrows=200, encoding="utf-8-sig")
            safe_dataframe(df.head(200))
        except Exception as exc:
            st.warning(f"预览失败：{exc}")


def download_file(path: Path, label: str):
    if path and path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
    )


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def download_dataframe_excel(df: pd.DataFrame, file_name: str, label: str, key: str = None):
    if df is None:
        return
    data = dataframe_to_excel_bytes(df)
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def reset_downstream(from_step):
    if from_step not in STEP_ORDER:
        return
    from_index = STEP_ORDER.index(from_step)
    downstream = STEP_ORDER[from_index + 1:]
    for step in downstream:
        st.session_state.step_done.pop(step, None)
        st.session_state.logs.pop(step, None)
    if from_step in ["merge", "dedup"]:
        st.session_state.logs.pop("update_ref", None)
    for key in [
        "dedup",
        "filtered",
        "processed",
        "processed_excluded",
        "high_iou",
        "other",
        "label_replaced",
        "label_replace_summary",
        "label_replace_diff",
        "label_replace_unmatched",
        "label_replace_sample_diff",
        "split_dir",
        "split_counts",
        "unclassified",
        "unclassified_summary",
        "category_files",
        "yolo_dir",
        "yolo_datasets",
        "yolo_skipped",
        "yolo_stats",
        "yolo_progress",
        "download_dir",
        "annotated_dir",
    ]:
        if from_step == "merge":
            st.session_state.outputs.pop(key, None)
        if from_step == "dedup" and key in ["filtered", "processed", "processed_excluded", "high_iou", "other", "download_dir", "annotated_dir", "split_dir", "split_counts", "unclassified", "unclassified_summary", "category_files", "yolo_dir", "yolo_datasets", "yolo_skipped"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "ref_filter" and key in ["processed", "processed_excluded", "high_iou", "other", "split_dir", "split_counts", "unclassified", "unclassified_summary", "category_files", "download_dir", "annotated_dir"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "replace_ptlist" and key in ["high_iou", "other", "label_replaced", "label_replace_summary", "label_replace_diff", "label_replace_unmatched", "label_replace_sample_diff", "split_dir", "split_counts", "unclassified", "unclassified_summary", "category_files", "download_dir", "annotated_dir"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "iou_filter" and key in ["label_replaced", "label_replace_summary", "label_replace_diff", "label_replace_unmatched", "label_replace_sample_diff", "split_dir", "split_counts", "unclassified", "unclassified_summary", "category_files", "download_dir", "annotated_dir"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "label_replace" and key in ["split_dir", "split_counts", "unclassified", "unclassified_summary", "category_files", "download_dir", "annotated_dir"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "split" and key in ["yolo_dir", "yolo_datasets", "yolo_skipped", "yolo_stats", "yolo_progress", "download_dir", "annotated_dir"]:
            st.session_state.outputs.pop(key, None)
        if from_step == "yolo" and key in ["download_dir", "annotated_dir", "yolo_stats", "yolo_progress"]:
            st.session_state.outputs.pop(key, None)


def step_status_chip(step_key, label):
    if st.session_state.step_done.get(step_key):
        chip_class = "chip-done"
        status = "已完成"
    else:
        chip_class = "chip-wait"
        status = "待执行"
    return f"<span class=\"chip {chip_class}\">{label} · {status}</span>"


def build_steps(config):
    label_enabled = bool(st.session_state.outputs.get("label_map_path"))
    return [
        ("merge", "合并CSV", False, True),
        ("dedup", "按source去重", False, True),
        ("ref_filter", "参考CSV去重", True, config.get("use_reference")),
        ("replace_ptlist", "替换ptList", False, True),
        ("iou_filter", "IoU筛选", False, True),
        ("label_replace", "标签替换", True, label_enabled),
        ("split", "规则分类拆分", False, True),
        ("yolo", "生成YOLO数据集", False, True),
        ("download", "下载并绘制标注", True, config.get("run_download")),
    ]


def render_stepper(config):
    steps = build_steps(config)
    html = "<div class='stepper'>"
    ready = True
    for idx, (key, label, optional, enabled) in enumerate(steps):
        if optional and not enabled:
            status = "skipped"
        elif st.session_state.step_done.get(key):
            status = "done"
        else:
            status = "active" if ready else "locked"
            ready = False
        if status == "done":
            ready = True
        if status == "skipped":
            ready = True
        html += (
            "<div class='step'>"
            f"<div class='step-circle {status}'>{idx + 1}</div>"
            f"<div class='step-label'>{label}</div>"
            "</div>"
        )
        if idx < len(steps) - 1:
            if status == "done":
                line_class = "line-done"
            elif status == "skipped":
                line_class = "line-skip"
            else:
                line_class = "line-lock"
            html += f"<div class='step-line {line_class}'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_dependency_graph(config):
    steps = build_steps(config)
    width = 160 * len(steps) + 40
    height = 120
    rect_w = 130
    rect_h = 40
    start_x = 20
    y = 30
    parts = [
        f"<svg class='dependency-card' viewBox='0 0 {width} {height}' width='100%' height='120' xmlns='http://www.w3.org/2000/svg'>",
        "<defs>",
        "<marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto'>",
        "<path d='M0,0 L10,3 L0,6 Z' fill='#94a3b8'></path>",
        "</marker>",
        "</defs>",
    ]

    for idx, (_, label, optional, enabled) in enumerate(steps):
        x = start_x + idx * 160
        stroke = "#2563eb" if enabled or not optional else "#cbd5f5"
        fill = "#ffffff" if enabled or not optional else "#f8fafc"
        dash = "" if enabled or not optional else "stroke-dasharray='4 4'"
        parts.append(
            f"<rect x='{x}' y='{y}' width='{rect_w}' height='{rect_h}' rx='10' fill='{fill}' stroke='{stroke}' {dash} />"
        )
        parts.append(
            f"<text x='{x + rect_w / 2}' y='{y + 24}' text-anchor='middle' font-size='12' fill='#0f172a'>{label}</text>"
        )
        if idx < len(steps) - 1:
            next_optional = steps[idx + 1][2]
            next_enabled = steps[idx + 1][3]
            dashed = (optional and not enabled) or (next_optional and not next_enabled)
            line_color = "#94a3b8" if dashed else "#2563eb"
            dash_attr = "stroke-dasharray='4 4'" if dashed else ""
            x1 = x + rect_w
            x2 = start_x + (idx + 1) * 160
            parts.append(
                f"<line x1='{x1}' y1='{y + rect_h / 2}' x2='{x2}' y2='{y + rect_h / 2}' stroke='{line_color}' stroke-width='2' marker-end='url(#arrow)' {dash_attr} />"
            )

    parts.append("</svg>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def compute_progress(config):
    active = ["merge", "dedup", "replace_ptlist", "iou_filter", "split", "yolo"]
    if config.get("use_reference"):
        active.insert(2, "ref_filter")
    if st.session_state.outputs.get("label_map_path"):
        active.insert(active.index("split"), "label_replace")
    if config.get("run_download"):
        active.append("download")
    done = sum(1 for s in active if st.session_state.step_done.get(s))
    total = len(active) if active else 1
    return done, total


@st.cache_data(show_spinner=False)
def get_row_count_cached(path_str, mtime):
    try:
        path_lower = str(path_str).lower()
        if path_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path_str)
            return len(df)
        if path_lower.endswith(".csv"):
            # Fast, low-memory line count to avoid reading large CSVs into pandas.
            line_count = 0
            with open(path_str, "r", encoding="utf-8-sig", errors="ignore") as f:
                for _ in f:
                    line_count += 1
            return max(line_count - 1, 0)
        df = pd.read_csv(path_str, encoding="utf-8-sig")
        return len(df)
    except Exception:
        return None


def get_row_count(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return get_row_count_cached(str(p), p.stat().st_mtime)
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


def summarize_yolo_label_counts(dataset_dirs):
    """统计每个YOLO数据集中 train/val/test 各标签“图片数量/标注框数量/占比”，并提供汇总。"""
    stats = {}
    flat_rows = []
    for dataset_dir in dataset_dirs or []:
        if not dataset_dir:
            continue
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            continue
        names = []
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                names = data.get("names") or []
            except Exception:
                names = []
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
                    except Exception:
                        continue
                    labels_in_image = set()
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except Exception:
                            continue
                        label_name = names[class_id] if class_id < len(names) else str(class_id)
                        labels_in_image.add(label_name)
                        box_counts[label_name] = box_counts.get(label_name, 0) + 1
                    for label in labels_in_image:
                        img_counts[label] = img_counts.get(label, 0) + 1
            split_stats[split] = {
                "total_images": total_images,
                "label_counts": img_counts,
                "box_counts": box_counts,
            }
            total_images_all += total_images
            for label, count in img_counts.items():
                total_img_counts[label] = total_img_counts.get(label, 0) + count
            for label, count in box_counts.items():
                total_box_counts[label] = total_box_counts.get(label, 0) + count

            all_labels = set(img_counts) | set(box_counts)
            for label in all_labels:
                img_count = img_counts.get(label, 0)
                box_count = box_counts.get(label, 0)
                ratio = (img_count / total_images) if total_images else 0.0
                flat_rows.append({
                    "数据集": dataset_key,
                    "split": split,
                    "标签": label,
                    "图片数量": img_count,
                    "标注框数量": box_count,
                    "占比%": f"{ratio * 100:.1f}%",
                    "split总图片数": total_images,
                })

        split_stats["all"] = {
            "total_images": total_images_all,
            "label_counts": total_img_counts,
            "box_counts": total_box_counts,
        }
        all_labels = set(total_img_counts) | set(total_box_counts)
        for label in all_labels:
            img_count = total_img_counts.get(label, 0)
            box_count = total_box_counts.get(label, 0)
            ratio = (img_count / total_images_all) if total_images_all else 0.0
            flat_rows.append({
                "数据集": dataset_key,
                "split": "all",
                "标签": label,
                "图片数量": img_count,
                "标注框数量": box_count,
                "占比%": f"{ratio * 100:.1f}%",
                "split总图片数": total_images_all,
            })

        stats[dataset_key] = split_stats
    df = pd.DataFrame(flat_rows)
    return stats, df


def format_int(value):
    return "-" if value is None else f"{value:,}"


def format_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{(numerator / denominator) * 100:.1f}%"


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


def render_merge_eta_card(elapsed, eta, speed, total_bytes, read_bytes, file_idx, total_files):
    progress = min(read_bytes / total_bytes, 1.0) if total_bytes else 0.0
    html = f"""
    <div class="glow-frame">
      <div class="glow-inner">
        <div class="kpi">合并进度</div>
        <div style="font-size:1.1rem;font-weight:700;">{progress*100:.1f}% · {file_idx}/{total_files} 文件</div>
        <div class="kpi" style="margin-top:8px;">速度 / 剩余</div>
        <div style="font-size:0.95rem;">{format_bytes(int(speed))}/s · 预计剩余 {format_duration(eta)}</div>
        <div class="kpi" style="margin-top:8px;">已用时</div>
        <div style="font-size:0.95rem;">{format_duration(elapsed)}</div>
      </div>
    </div>
    """
    return html


def render_stats_cards(items):
    if not items:
        return
    cards = ""
    for label, value, hint in items:
        hint_html = f"<div class='stat-hint'>{hint}</div>" if hint else ""
        cards += (
            "<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            f"{hint_html}"
            "</div>"
        )
    st.markdown(f"<div class='stat-grid'>{cards}</div>", unsafe_allow_html=True)


def collect_counts(outputs):
    return {
        "merged": get_row_count(outputs.get("merged")),
        "dedup": get_row_count(outputs.get("dedup")),
        "filtered": get_row_count(outputs.get("filtered")),
        "processed": get_row_count(outputs.get("processed")),
        "processed_excluded": get_row_count(outputs.get("processed_excluded")),
        "high_iou": get_row_count(outputs.get("high_iou")),
        "other": get_row_count(outputs.get("other")),
        "label_replaced": get_row_count(outputs.get("label_replaced")),
        "unclassified": get_row_count(outputs.get("unclassified")),
        "unclassified_summary": get_row_count(outputs.get("unclassified_summary")),
        "split_counts": get_row_count(outputs.get("split_counts")),
    }


def get_summary_metrics(counts):
    total = counts.get("merged")
    processed = counts.get("processed")
    high_iou = counts.get("high_iou")
    other = counts.get("other")
    final_total = None
    if high_iou is not None and other is not None:
        final_total = high_iou + other
    final_retention = format_ratio(final_total, total)
    hit_rate = format_ratio(high_iou, processed)
    return [
        ("最终输出行数", format_int(final_total), "高IoU + 其他"),
        ("最终保留率", final_retention, "最终输出/合并结果"),
        ("高IoU命中率", hit_rate, "高IoU/ptList替换结果"),
    ]


def list_excel_files(folder_path):
    if not folder_path:
        return []
    folder = Path(folder_path)
    if not folder.exists():
        return []
    files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    return sorted(files)


def build_export_zip(outputs, include_images=False, only_classification=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if not only_classification:
            csv_keys = [
                "merged",
                "dedup",
                "filtered",
                "processed",
                "high_iou",
                "other",
                "label_replaced",
                "label_replace_diff",
                "label_replace_unmatched",
                "unclassified",
                "unclassified_summary",
                "split_counts",
            ]
            for key in csv_keys:
                path = outputs.get(key)
                if path and Path(path).exists():
                    zf.write(path, arcname=f"csv/{Path(path).name}")
        else:
            path = outputs.get("unclassified")
            if path and Path(path).exists():
                zf.write(path, arcname=f"categories/{Path(path).name}")
            path = outputs.get("unclassified_summary")
            if path and Path(path).exists():
                zf.write(path, arcname=f"categories/{Path(path).name}")
            path = outputs.get("split_counts")
            if path and Path(path).exists():
                zf.write(path, arcname=f"categories/{Path(path).name}")

        category_files = outputs.get("category_files") or []
        for path in category_files:
            if path and Path(path).exists():
                zf.write(path, arcname=f"categories/{Path(path).name}")

        if include_images:
            annotated_dir = outputs.get("annotated_dir")
            download_dir = outputs.get("download_dir")
            for folder, prefix in [(download_dir, "images/downloaded"), (annotated_dir, "images/annotated")]:
                if folder and Path(folder).exists():
                    for file_path in Path(folder).glob("*"):
                        if file_path.is_file():
                            zf.write(file_path, arcname=f"{prefix}/{file_path.name}")

    buffer.seek(0)
    return buffer


def build_yolo_zip(yolo_dir):
    if not yolo_dir:
        return None
    yolo_dir = Path(yolo_dir)
    if not yolo_dir.exists():
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in yolo_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(yolo_dir)))
    buffer.seek(0)
    return buffer


def ensure_empty_reference_csv(path_str, template_csv_path=None):
    if not path_str:
        return False, "参考CSV路径为空"
    path = Path(path_str)
    if path.exists():
        return True, None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["source"]
        if template_csv_path and Path(template_csv_path).exists():
            try:
                columns = list(pd.read_csv(template_csv_path, nrows=0, encoding="utf-8-sig").columns)
            except Exception:
                columns = ["source"]
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
        return True, f"已自动创建空参考文件（继承主CSV列）：{path}"
    except Exception as exc:
        return False, f"自动创建参考文件失败：{exc}"


def render_output_preview(outputs):
    preview_items = [
        ("合并结果", outputs.get("merged")),
        ("去重结果", outputs.get("dedup")),
        ("参考去重结果", outputs.get("filtered")),
        ("ptList替换结果", outputs.get("processed")),
        ("ptList未筛选", outputs.get("processed_excluded")),
        ("高IoU结果", outputs.get("high_iou")),
        ("其他数据", outputs.get("other")),
        ("标签替换结果", outputs.get("label_replaced")),
        ("标签替换差异", outputs.get("label_replace_diff")),
        ("标签替换未匹配", outputs.get("label_replace_unmatched")),
        ("无法分类数据", outputs.get("unclassified")),
        ("无法分类汇总", outputs.get("unclassified_summary")),
        ("拆分条数统计", outputs.get("split_counts")),
        ("YOLO跳过清单", outputs.get("yolo_skipped")),
    ]
    available = [(label, path) for label, path in preview_items if path and Path(path).exists()]
    if not available:
        st.info("暂无可预览的输出文件。")
        return

    st.markdown("**输出预览（点击查看）**")
    cols = st.columns(2)
    with cols[0]:
        for label, path in available[: (len(available) + 1) // 2]:
            if st.button(f"预览 {label}", key=f"preview_{label}"):
                st.session_state.preview_path = path
    with cols[1]:
        for label, path in available[(len(available) + 1) // 2:]:
            if st.button(f"预览 {label}", key=f"preview_{label}"):
                st.session_state.preview_path = path

    if st.session_state.preview_path:
        preview_csv(Path(st.session_state.preview_path), "当前预览")


existing_input_csvs = []
ref_fallback_path = None
rule_fallback_path = None
label_map_fallback_path = None

with st.sidebar:
    st.markdown("<div class='sidebar-title'>配置中心</div>", unsafe_allow_html=True)
    st.caption("输出目录（固定，覆盖旧结果）")
    st.code(str(FIXED_OUTPUT_ROOT))

    uploaded_csvs = st.file_uploader(
        "上传待处理CSV（支持多文件）",
        type=["csv"],
        accept_multiple_files=True,
    )
    input_dir_default = FIXED_OUTPUT_ROOT / "input_csvs"
    if not uploaded_csvs and input_dir_default.exists():
        existing_input_csvs = sorted(input_dir_default.glob("*.csv"))

    if uploaded_csvs:
        render_file_tiles("已上传主CSV", [file_info_from_upload(f) for f in uploaded_csvs], columns=4)
    elif existing_input_csvs:
        render_file_tiles("已保存主CSV", [file_info_from_path(p) for p in existing_input_csvs], columns=4)
        st.caption("未重新上传，默认使用已保存的主CSV文件。")
        if st.button("删除历史主CSV", key="clear_saved_csvs", width='stretch'):
            st.session_state["confirm_clear_saved_csvs"] = True
        if st.session_state.get("confirm_clear_saved_csvs"):
            keep_files = []
            for name in ["reference.csv", "classification_rules.xlsx", "label_mapping.xlsx"]:
                if (FIXED_OUTPUT_ROOT / name).exists():
                    keep_files.append(name)
            def _do_clear_csvs():
                try:
                    for p in existing_input_csvs:
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    try:
                        input_dir_default.rmdir()
                    except Exception:
                        pass
                    clear_output_root(FIXED_OUTPUT_ROOT, keep_inputs=False, keep_files=keep_files)
                    if st.session_state.outputs.get("input_dir") and Path(st.session_state.outputs.get("input_dir")) == input_dir_default:
                        st.session_state.input_ready = False
                        st.session_state.outputs["uploaded_info"] = []
                        st.session_state.outputs["input_dir"] = str(input_dir_default)
                        st.session_state.step_done = {}
                        st.session_state.logs = {}
                    st.success("已删除历史上传主CSV，并清理相关输出。")
                except Exception as exc:
                    st.error(f"删除历史主CSV失败：{exc}")
            show_confirm_dialog(
                "confirm_clear_saved_csvs",
                "确认删除历史主CSV",
                "将删除已保存的主CSV文件，并清理 runs/latest 下的相关输出。此操作不可恢复。",
                _do_clear_csvs,
            )

    sample_csv = None
    if uploaded_csvs:
        sample_csv = uploaded_csvs[0]
    elif existing_input_csvs:
        sample_csv = existing_input_csvs[0]
    if sample_csv:
        cols = get_csv_columns(sample_csv)
        if cols is not None:
            required_cols = ["source", "是否废弃", "结果字段-目标检测标签配置"]
            missing_cols = [c for c in required_cols if c not in cols]
            if missing_cols:
                st.warning(f"主CSV缺少必要列：{', '.join(missing_cols)}")
        else:
            st.info("主CSV列读取失败，请确认文件编码或格式。")

    use_reference = st.checkbox("启用参考CSV去重", value=True)
    ref_mode = st.radio("参考CSV来源", ["上传参考CSV", "使用已有路径"], horizontal=True)
    ref_path = None
    ref_uploaded = None

    if ref_mode == "上传参考CSV":
        ref_uploaded = st.file_uploader("上传参考CSV", type=["csv"], key="ref_csv")
        if ref_uploaded:
            render_file_tiles("已上传参考CSV", [file_info_from_upload(ref_uploaded)])
        else:
            candidate = FIXED_OUTPUT_ROOT / "reference.csv"
            if candidate.exists():
                ref_fallback_path = candidate
                render_file_tiles("已保存参考CSV", [file_info_from_path(candidate)])
                st.caption(f"未重新上传，默认使用：{candidate.name}")
                if st.button("删除参考CSV", key="clear_ref_csv", width='stretch'):
                    st.session_state["confirm_clear_ref"] = True
                if st.session_state.get("confirm_clear_ref"):
                    keep_files = []
                    for name in ["classification_rules.xlsx", "label_mapping.xlsx"]:
                        if (FIXED_OUTPUT_ROOT / name).exists():
                            keep_files.append(name)
                    def _do_clear_ref():
                        try:
                            candidate.unlink(missing_ok=True)
                            clear_output_root(FIXED_OUTPUT_ROOT, keep_inputs=True, keep_files=keep_files)
                            st.session_state.outputs["ref_path"] = None
                            st.session_state.outputs["ref_info"] = []
                            st.session_state.input_ready = False
                            st.session_state.step_done = {}
                            st.session_state.logs = {}
                            st.success("已删除参考CSV，并清理相关输出。")
                        except Exception as exc:
                            st.error(f"删除参考CSV失败：{exc}")
                    show_confirm_dialog(
                        "confirm_clear_ref",
                        "确认删除参考CSV",
                        "将删除参考CSV文件，并清理 runs/latest 下的相关输出。此操作不可恢复。",
                        _do_clear_ref,
                    )
        if ref_uploaded and use_reference:
            ref_cols = get_csv_columns(ref_uploaded)
            if ref_cols is not None and "source" not in ref_cols:
                st.warning("参考CSV缺少 source 列，去重可能失败。")
    else:
        ref_path = st.text_input("参考CSV路径", value=str(Path.cwd() / "reference.csv"))
        if use_reference and ref_path and not Path(ref_path).exists():
            st.info("参考CSV路径不存在，将在确认输入时自动创建空文件。")
        if use_reference and ref_path and Path(ref_path).exists():
            ref_cols = get_csv_columns(ref_path)
            if ref_cols is not None and "source" not in ref_cols:
                st.warning("参考CSV缺少 source 列，去重可能失败。")

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>合并设置</div>", unsafe_allow_html=True)
    merge_chunk_size = st.number_input("合并分块行数", min_value=1000, max_value=500000, value=100000, step=1000)
    keep_outputs = st.checkbox("保留旧输出用于跳过", value=True)

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>处理参数</div>", unsafe_allow_html=True)
    min_boxes = st.number_input("最小标注框数量", min_value=1, max_value=50, value=2, step=1)
    iou_threshold = st.number_input("IoU阈值", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
    update_reference = st.checkbox("覆盖更新reference.csv", value=False)
    backup_reference = st.checkbox("更新时备份reference.csv", value=True)

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>分类规则</div>", unsafe_allow_html=True)
    rule_source = st.radio("规则来源", ["上传规则Excel", "指定文件夹"], horizontal=True)
    rule_upload = None
    rule_folder = None
    rule_file_path = None

    if rule_source == "上传规则Excel":
        rule_upload = st.file_uploader("上传分类规则Excel", type=["xlsx", "xls"], key="rule_excel")
        if rule_upload:
            st.caption(f"已选择：{rule_upload.name}")
        else:
            candidate = FIXED_OUTPUT_ROOT / "classification_rules.xlsx"
            if candidate.exists():
                rule_fallback_path = candidate
                st.caption(f"未重新上传，默认使用：{candidate.name}")
                if st.button("删除规则文件", key="clear_rule_excel", width='stretch'):
                    st.session_state["confirm_clear_rule"] = True
                if st.session_state.get("confirm_clear_rule"):
                    keep_files = []
                    for name in ["reference.csv", "label_mapping.xlsx"]:
                        if (FIXED_OUTPUT_ROOT / name).exists():
                            keep_files.append(name)
                    def _do_clear_rule():
                        try:
                            candidate.unlink(missing_ok=True)
                            clear_output_root(FIXED_OUTPUT_ROOT, keep_inputs=True, keep_files=keep_files)
                            st.session_state.outputs["rule_path"] = None
                            st.session_state.input_ready = False
                            st.session_state.step_done = {}
                            st.session_state.logs = {}
                            st.success("已删除规则文件，并清理相关输出。")
                        except Exception as exc:
                            st.error(f"删除规则文件失败：{exc}")
                    show_confirm_dialog(
                        "confirm_clear_rule",
                        "确认删除规则文件",
                        "将删除分类规则文件，并清理 runs/latest 下的相关输出。此操作不可恢复。",
                        _do_clear_rule,
                    )
    else:
        rule_folder = st.text_input("规则文件夹路径", value=str(Path.cwd()))
        excel_files = list_excel_files(rule_folder)
        if not excel_files:
            st.info("该目录未找到Excel文件。")
        else:
            rule_file_path = st.selectbox(
                "选择规则文件",
                options=[str(p) for p in excel_files],
                format_func=lambda x: Path(x).name,
            )

    rule_mode = st.radio("解析方式", ["宽表(类别为列)", "两列映射"], horizontal=True)
    st.caption("宽表模式：每列是大类，每格是子标签；支持逗号/顿号/分号/换行分隔，标签中的“|”不会拆分。")
    rule_sheet = None
    rule_label_col = None
    rule_category_col = None
    rule_columns = []

    if rule_source == "上传规则Excel":
        rules_source_obj = rule_upload if rule_upload else rule_fallback_path
    else:
        rules_source_obj = rule_file_path if rule_file_path else None
    if rules_source_obj:
        try:
            excel_file = pd.ExcelFile(rules_source_obj)
            rule_sheet = st.selectbox("规则Sheet", options=excel_file.sheet_names)
            preview_df = excel_file.parse(rule_sheet, nrows=5)
            rule_columns = list(preview_df.columns)
            if rule_mode == "两列映射" and rule_columns:
                rule_label_col = st.selectbox("标签列", options=rule_columns, index=0)
                rule_category_col = st.selectbox(
                    "类别列",
                    options=rule_columns,
                    index=1 if len(rule_columns) > 1 else 0,
                )
            with st.expander("规则预览（前50行）", expanded=False):
                preview_full = excel_file.parse(rule_sheet, nrows=50)
                safe_dataframe(preview_full, width='stretch')
        except Exception as exc:
            st.warning(f"规则文件读取失败：{exc}")
    train_ratio = st.number_input("训练集比例", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    val_ratio = st.number_input("验证集比例", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    test_ratio = st.number_input("测试集比例", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    random_seed = st.number_input("拆分随机种子", min_value=0, max_value=9999, value=42, step=1)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        st.warning("训练/验证/测试比例之和必须大于0。")
    elif abs(ratio_sum - 1.0) > 0.01:
        st.info("比例之和不为1，将在执行时自动归一化。")

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>标签替换</div>", unsafe_allow_html=True)
    label_map_upload = st.file_uploader("上传新旧标签对照表Excel", type=["xlsx", "xls"], key="label_map_excel")
    label_map_sheet = None
    label_map_old_col = None
    label_map_new_col = None
    if label_map_upload is None:
        candidate = FIXED_OUTPUT_ROOT / "label_mapping.xlsx"
        if candidate.exists():
            label_map_fallback_path = candidate
            st.caption(f"未重新上传，默认使用：{candidate.name}")
            if st.button("删除标签对照表", key="clear_label_map", width='stretch'):
                st.session_state["confirm_clear_label"] = True
            if st.session_state.get("confirm_clear_label"):
                keep_files = []
                for name in ["reference.csv", "classification_rules.xlsx"]:
                    if (FIXED_OUTPUT_ROOT / name).exists():
                        keep_files.append(name)
                def _do_clear_label():
                    try:
                        candidate.unlink(missing_ok=True)
                        clear_output_root(FIXED_OUTPUT_ROOT, keep_inputs=True, keep_files=keep_files)
                        st.session_state.outputs["label_map_path"] = None
                        st.session_state.input_ready = False
                        st.session_state.step_done = {}
                        st.session_state.logs = {}
                        st.success("已删除标签对照表，并清理相关输出。")
                    except Exception as exc:
                        st.error(f"删除标签对照表失败：{exc}")
                show_confirm_dialog(
                    "confirm_clear_label",
                    "确认删除标签对照表",
                    "将删除标签对照表文件，并清理 runs/latest 下的相关输出。此操作不可恢复。",
                    _do_clear_label,
                )
    label_map_source_obj = label_map_upload if label_map_upload else label_map_fallback_path
    if label_map_source_obj:
        try:
            label_excel = pd.ExcelFile(label_map_source_obj)
            label_map_sheet = st.selectbox("对照表Sheet", options=label_excel.sheet_names, key="label_map_sheet")
            preview_df = label_excel.parse(label_map_sheet, nrows=5)
            map_columns = list(preview_df.columns)
            if map_columns:
                label_map_old_col = st.selectbox("旧标签列", options=map_columns, index=0, key="label_map_old_col")
                label_map_new_col = st.selectbox(
                    "新标签列",
                    options=map_columns,
                    index=1 if len(map_columns) > 1 else 0,
                    key="label_map_new_col",
                )
            with st.expander("对照表预览（前50行）", expanded=False):
                preview_full = label_excel.parse(label_map_sheet, nrows=50)
                safe_dataframe(preview_full, width='stretch')
        except Exception as exc:
            st.warning(f"标签对照表读取失败：{exc}")

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>标注输出</div>", unsafe_allow_html=True)
    run_download = st.checkbox("下载并绘制标注图片", value=False)
    max_images = st.number_input("最多处理图片数（0表示不限）", min_value=0, max_value=100000, value=0, step=10)
    max_images = None if max_images == 0 else int(max_images)

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>YOLO设置</div>", unsafe_allow_html=True)
    class_order_text = st.text_area("YOLO类顺序（每行一个标签）", value="", height=120)
    st.caption("留空则按类别内标签字母序；填写后会优先按此顺序生成 class id。")

    confirm_input = st.button("确认输入并保存", width='stretch')

if confirm_input:
    has_uploaded_csvs = bool(uploaded_csvs)
    has_existing_csvs = bool(existing_input_csvs)
    if not has_uploaded_csvs and not has_existing_csvs:
        st.error("请先上传至少一个CSV文件，或保留输出目录中已保存的输入CSV。")
    elif use_reference and ref_mode == "上传参考CSV" and ref_uploaded is None and ref_fallback_path is None:
        st.error("已启用参考CSV去重，请上传参考CSV或保留已有 reference.csv。")
    elif use_reference and ref_mode == "使用已有路径" and not ref_path:
        st.error("参考CSV路径为空，请提供有效路径或改为上传参考CSV。")
    else:
        st.session_state.output_root = str(FIXED_OUTPUT_ROOT)
        output_root_path = FIXED_OUTPUT_ROOT
        if output_root_path.exists() and not keep_outputs:
            try:
                shutil.rmtree(output_root_path)
            except Exception as exc:
                st.error(f"清理旧输出失败：{exc}")
                st.stop()
        output_root_path.mkdir(parents=True, exist_ok=True)
        input_dir = output_root_path / "input_csvs"
        if has_uploaded_csvs:
            if input_dir.exists():
                try:
                    shutil.rmtree(input_dir)
                except Exception as exc:
                    st.error(f"清理旧输入失败：{exc}")
                    st.stop()
            saved_csvs = save_uploads(uploaded_csvs, input_dir)
            st.success(f"已保存 {len(saved_csvs)} 个CSV到：{input_dir}")
        else:
            input_dir.mkdir(parents=True, exist_ok=True)
            saved_csvs = existing_input_csvs
            st.success(f"使用已保存 {len(saved_csvs)} 个CSV：{input_dir}")

        ref_path_value = None
        if use_reference:
            if ref_mode == "上传参考CSV":
                if ref_uploaded:
                    ref_path_value = output_root_path / "reference.csv"
                    save_upload(ref_uploaded, ref_path_value)
                else:
                    ref_path_value = ref_fallback_path
            else:
                ref_path_value = Path(ref_path)
                template_csv = str(saved_csvs[0]) if saved_csvs else None
                ok, msg = ensure_empty_reference_csv(str(ref_path_value), template_csv)
                if not ok:
                    st.error(msg)
                    st.stop()
                if msg:
                    st.success(msg)
        st.session_state.outputs["ref_path"] = ref_path_value
        st.session_state.outputs["input_dir"] = input_dir
        if has_uploaded_csvs:
            st.session_state.outputs["uploaded_info"] = [file_info_from_upload(f) for f in uploaded_csvs]
        else:
            st.session_state.outputs["uploaded_info"] = [file_info_from_path(p) for p in existing_input_csvs]
        if ref_uploaded:
            st.session_state.outputs["ref_info"] = [file_info_from_upload(ref_uploaded)]
        elif ref_path_value and Path(ref_path_value).exists():
            st.session_state.outputs["ref_info"] = [file_info_from_path(Path(ref_path_value))]
        else:
            st.session_state.outputs["ref_info"] = []
        label_map_path_value = None
        if label_map_upload is not None:
            label_map_path_value = output_root_path / "label_mapping.xlsx"
            save_upload(label_map_upload, label_map_path_value)
        elif label_map_fallback_path:
            label_map_path_value = label_map_fallback_path
        st.session_state.outputs["label_map_path"] = label_map_path_value
        rule_path_value = None
        if rule_source == "上传规则Excel" and rule_upload is not None:
            rule_path_value = output_root_path / "classification_rules.xlsx"
            save_upload(rule_upload, rule_path_value)
        elif rule_source == "上传规则Excel" and rule_fallback_path:
            rule_path_value = rule_fallback_path
        elif rule_source == "指定文件夹" and rule_file_path:
            rule_path_value = Path(rule_file_path)
        st.session_state.outputs["rule_path"] = rule_path_value
        st.session_state.input_ready = True
        st.session_state.step_done = {}
        st.session_state.logs = {}
        st.session_state.config = {
            "use_reference": use_reference,
            "update_reference": update_reference,
            "backup_reference": backup_reference,
            "merge_chunk_size": int(merge_chunk_size),
            "keep_outputs": bool(keep_outputs),
            "min_boxes": int(min_boxes),
            "iou_threshold": float(iou_threshold),
            "run_download": run_download,
            "max_images": max_images,
            "ref_mode": ref_mode,
            "rule_mode": rule_mode,
            "rule_sheet": rule_sheet,
            "rule_label_col": rule_label_col,
            "rule_category_col": rule_category_col,
            "label_map_sheet": label_map_sheet,
            "label_map_old_col": label_map_old_col,
            "label_map_new_col": label_map_new_col,
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "random_seed": int(random_seed),
            "class_order": [line.strip() for line in class_order_text.splitlines() if line.strip()],
        }

if not st.session_state.input_ready:
    st.info("请在左侧配置区完成输入并点击“确认输入并保存”。")
    st.stop()
else:
    st.caption("如需修改输入或参数，请重新点击“确认输入并保存”。")

output_root_path = Path(st.session_state.output_root)
config = st.session_state.config
counts = collect_counts(st.session_state.outputs)

st.markdown("---")

st.markdown("**运行概览**")
summary_left, summary_right = st.columns([2, 1])
with summary_left:
    st.markdown(
        f"""
        <div class="glow-frame">
          <div class="glow-inner">
            <div class="kpi">运行ID</div>
            <div style="font-size:1.1rem;font-weight:700;">{st.session_state.run_id}</div>
            <div class="kpi" style="margin-top:8px;">输出目录</div>
            <div style="font-size:0.9rem;">{output_root_path}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_right:
    done, total = compute_progress(config)
    st.markdown(
        f"""
        <div class="glow-frame">
          <div class="glow-inner">
            <div class="kpi">流程进度</div>
            <div style="font-size:1.1rem;font-weight:700;">{done} / {total}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(done / total)

st.markdown("**步骤进度条**")
render_stepper(config)

st.markdown("**流程依赖图**")
render_dependency_graph(config)

st.markdown("**结果指标总览**")
render_stats_cards(get_summary_metrics(counts))

render_output_preview(st.session_state.outputs)

if st.session_state.outputs.get("uploaded_info"):
    st.markdown("**输入文件**")
    render_file_tiles("主CSV", st.session_state.outputs.get("uploaded_info", []), columns=4)
    if config.get("use_reference"):
        render_file_tiles("参考CSV", st.session_state.outputs.get("ref_info", []), columns=4)

st.markdown("**分类规则 Excel 预览**")
rule_path = st.session_state.outputs.get("rule_path")
if rule_path and Path(rule_path).exists():
    try:
        rule_sheet = config.get("rule_sheet")
        preview_rules = pd.read_excel(rule_path, sheet_name=rule_sheet, nrows=200) if rule_sheet else pd.read_excel(rule_path, nrows=200)
        safe_dataframe(preview_rules)
    except Exception as exc:
        st.warning(f"规则预览失败：{exc}")
else:
    st.info("尚未选择分类规则文件。")

st.markdown("---")


with st.expander("Step 1 合并CSV", expanded=True):
    st.markdown(step_status_chip("merge", "合并CSV"), unsafe_allow_html=True)
    input_dir_dbg = st.session_state.outputs.get("input_dir")
    input_dir_path_dbg = Path(input_dir_dbg) if input_dir_dbg else None
    input_csv_count_dbg = None
    if input_dir_path_dbg and input_dir_path_dbg.exists():
        input_csv_count_dbg = len(list(input_dir_path_dbg.glob("*.csv")))
    st.caption(
        f"调试提示：input_ready = {st.session_state.input_ready} | "
        f"input_dir = {input_dir_dbg or '-'} | "
        f"csv文件数 = {input_csv_count_dbg if input_csv_count_dbg is not None else '-'}"
    )
    merge_btn = st.button(
        "确认并执行 Step 1",
        disabled=not st.session_state.input_ready,
        key="run_merge",
        width='stretch',
    )
    if merge_btn:
        st.info("Step 1 已进入执行分支")
        reset_downstream("merge")
        merged_csv = output_root_path / "merged_result.csv"
        input_dir = Path(st.session_state.outputs["input_dir"])
        input_files = sorted(input_dir.glob("*.csv"))
        log_dir = output_root_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"merge_{st.session_state.run_id}.log"
        st.session_state.outputs["merge_log"] = log_path
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"==== 合并开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
                f.write("校验：已点击 Step 1 并进入执行分支。\n")
                f.write(f"input_dir: {input_dir}\n")
                f.write(f"input_csv_count: {len(input_files)}\n")
                if input_files:
                    preview_names = [p.name for p in input_files[:20]]
                    f.write(f"input_csv_preview: {', '.join(preview_names)}\n")
            st.info(f"合并已启动，日志路径：`{log_path}`")
        except Exception:
            pass

        skip_merge = False
        if merged_csv.exists() and input_files:
            latest_input = max(f.stat().st_mtime for f in input_files)
            if merged_csv.stat().st_mtime >= latest_input:
                st.info("检测到 merged_result.csv 已是最新，已跳过合并。")
                st.session_state.outputs["merged"] = merged_csv
                st.session_state.step_done["merge"] = True
                st.session_state.logs["merge"] = "快速跳过：merged_result.csv 已是最新。"
                counts["merged"] = get_row_count(merged_csv)
                skip_merge = True
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("快速跳过：merged_result.csv 已是最新。\n")
                except Exception:
                    pass

        if not skip_merge:
            progress_bar = st.progress(0.0)
            eta_card = st.empty()
            log_box = st.empty()
            heartbeat_box = st.empty()
            log_lines = []
            start_time = time.time()
            merge_state = {
                "rows": 0,
                "bytes": 0,
                "bytes_read": 0,
                "file_idx": 0,
                "files": len(input_files),
            }
            last_heartbeat = {"ts": 0.0}
            log_file = open(log_path, "a", encoding="utf-8")
            ui_tick = {"ts": 0.0}

            def _progress_cb(file_idx, total_files, filename, total_rows, file_rows, chunk_idx, file_size, file_bytes, total_bytes, total_bytes_read):
                if total_files:
                    progress_bar.progress(min(file_idx / total_files, 1.0))
                elapsed = max(time.time() - start_time, 0.001)
                speed = total_bytes_read / elapsed
                eta = (total_bytes - total_bytes_read) / speed if speed > 0 else None
                file_pct = (file_bytes / file_size * 100) if file_size else 0
                message = (
                    f"处理第 {file_idx}/{total_files} 个文件：{filename} | "
                    f"当前文件 {file_rows} 行 | 已合并 {total_rows} 行\\n"
                    f"当前文件大小 {format_bytes(file_size)} | 读取 {file_pct:.1f}% | "
                    f"速度 {format_bytes(int(speed))}/s | 预计剩余 {format_duration(eta)}"
                )
                log_lines.append(message)
                if len(log_lines) > 200:
                    log_lines[:] = log_lines[-200:]
                try:
                    log_file.write(message + "\\n\\n")
                    log_file.flush()
                except Exception:
                    pass
                if time.time() - ui_tick["ts"] >= 0.3 or chunk_idx == 1:
                    log_box.text_area("合并日志（实时）", "\\n\\n".join(log_lines), height=220)
                    ui_tick["ts"] = time.time()
                if time.time() - last_heartbeat["ts"] >= 2:
                    heartbeat_box.caption(f"心跳：仍在合并中… {datetime.now().strftime('%H:%M:%S')}")
                    last_heartbeat["ts"] = time.time()
                if time.time() - ui_tick["ts"] >= 0.3 or chunk_idx == 1:
                    eta_card.markdown(
                        render_merge_eta_card(
                            elapsed,
                            eta,
                            speed,
                            total_bytes,
                            total_bytes_read,
                            file_idx,
                            total_files,
                        ),
                        unsafe_allow_html=True,
                    )
                merge_state["rows"] = total_rows
                merge_state["bytes"] = total_bytes
                merge_state["bytes_read"] = total_bytes_read
                merge_state["file_idx"] = file_idx
                merge_state["files"] = total_files

            try:
                with st.spinner("合并中，请耐心等待…"):
                    merge_all_csv_in_folder(
                        str(input_dir),
                        str(merged_csv),
                        "utf-8-sig",
                        int(config.get("merge_chunk_size", 100000)),
                        _progress_cb,
                    )
            finally:
                try:
                    log_file.close()
                except Exception:
                    pass

            st.session_state.outputs["merged"] = merged_csv
            st.session_state.step_done["merge"] = True
            st.session_state.logs["merge"] = "\n".join(log_lines)
            counts["merged"] = get_row_count(merged_csv)
            st.success("合并CSV 完成")

            elapsed_total = max(time.time() - start_time, 0.001)
            avg_speed = merge_state["bytes_read"] / elapsed_total if elapsed_total else 0
            summary = {
                "files": merge_state["files"],
                "rows": merge_state["rows"],
                "bytes": merge_state["bytes"],
                "elapsed": elapsed_total,
                "avg_speed": avg_speed,
            }
            st.session_state.outputs["merge_summary"] = summary

            log_dir = output_root_path / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"merge_{st.session_state.run_id}.log"
            summary_lines = [
                "==== 合并总结 ====",
                f"文件数: {summary['files']}",
                f"总行数: {summary['rows']}",
                f"总大小: {format_bytes(int(summary['bytes']))}",
                f"耗时: {format_duration(summary['elapsed'])}",
                f"平均速度: {format_bytes(int(summary['avg_speed']))}/s",
            ]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("\n".join(summary_lines))
                f.write("\n")
            st.session_state.outputs["merge_log"] = log_path

    render_stats_cards([
        ("输入文件", format_int(len(st.session_state.outputs.get("uploaded_info", []))), "主CSV数量"),
        ("合并行数", format_int(counts.get("merged")), "输出行数"),
    ])

    merge_summary = st.session_state.outputs.get("merge_summary")
    if merge_summary:
        render_stats_cards([
            ("合并耗时", format_duration(merge_summary.get("elapsed")), "总耗时"),
            ("合并总大小", format_bytes(int(merge_summary.get("bytes") or 0)), "输入CSV总大小"),
            ("平均速度", f"{format_bytes(int(merge_summary.get('avg_speed') or 0))}/s", "平均吞吐"),
        ])
        merge_log = st.session_state.outputs.get("merge_log")
        if merge_log and Path(merge_log).exists():
            st.download_button(
                label="下载合并日志",
                data=Path(merge_log).read_bytes(),
                file_name=Path(merge_log).name,
                mime="text/plain",
                width='stretch',
            )

    show_logs("merge", "合并CSV")
    preview_csv(st.session_state.outputs.get("merged"), "合并结果")
    download_file(st.session_state.outputs.get("merged"), "下载 merged_result.csv")

with st.expander("Step 2 按source去重", expanded=False):
    st.markdown(step_status_chip("dedup", "按source去重"), unsafe_allow_html=True)
    dedup_btn = st.button(
        "确认并执行 Step 2",
        disabled=not st.session_state.step_done.get("merge"),
        key="run_dedup",
        width='stretch',
    )
    if dedup_btn:
        reset_downstream("dedup")
        dedup_csv = output_root_path / "deduplicate_result.csv"
        run_step("dedup", "按source去重", deduplicate_csv_by_source, str(st.session_state.outputs["merged"]), str(dedup_csv))
        st.session_state.outputs["dedup"] = dedup_csv
        st.session_state.step_done["dedup"] = True
        counts["dedup"] = get_row_count(dedup_csv)
        if not config.get("use_reference"):
            st.session_state.outputs["filtered"] = dedup_csv
            st.session_state.step_done["ref_filter"] = True

    removed = None
    if counts.get("merged") is not None and counts.get("dedup") is not None:
        removed = counts["merged"] - counts["dedup"]

    render_stats_cards([
        ("输入行数", format_int(counts.get("merged")), "合并结果"),
        ("去重后行数", format_int(counts.get("dedup")), "去重输出"),
        ("去除重复", format_int(removed), "减少行数"),
        ("保留率", format_ratio(counts.get("dedup"), counts.get("merged")), "去重后/去重前"),
    ])

    show_logs("dedup", "按source去重")
    preview_csv(st.session_state.outputs.get("dedup"), "去重结果")
    download_file(st.session_state.outputs.get("dedup"), "下载 deduplicate_result.csv")

with st.expander("Step 3 参考CSV去重", expanded=False):
    if config.get("use_reference"):
        st.markdown(step_status_chip("ref_filter", "参考CSV去重"), unsafe_allow_html=True)
        ref_exists = True
        ref_path_value = st.session_state.outputs.get("ref_path")
        if config.get("ref_mode") == "使用已有路径":
            ref_exists = bool(ref_path_value) and Path(ref_path_value).exists()
            if not ref_exists:
                st.warning("参考CSV路径不存在，请回到左侧配置区修正后重新确认输入。")
        ref_btn = st.button(
            "确认并执行 Step 3",
            disabled=not st.session_state.step_done.get("dedup") or not ref_exists,
            key="run_ref",
            width='stretch',
        )
        if ref_btn:
            reset_downstream("ref_filter")
            filtered_csv = output_root_path / "filtered_main.csv"
            run_step(
                "ref_filter",
                "参考CSV去重",
                remove_duplicates_between_csv,
                str(st.session_state.outputs["dedup"]),
                str(st.session_state.outputs["ref_path"]),
                str(filtered_csv),
            )
            st.session_state.outputs["filtered"] = filtered_csv
            st.session_state.step_done["ref_filter"] = True
            counts["filtered"] = get_row_count(filtered_csv)

        removed = None
        if counts.get("dedup") is not None and counts.get("filtered") is not None:
            removed = counts["dedup"] - counts["filtered"]

        render_stats_cards([
            ("输入行数", format_int(counts.get("dedup")), "去重结果"),
            ("过滤后行数", format_int(counts.get("filtered")), "参考去重输出"),
            ("剔除行数", format_int(removed), "与参考集重复"),
            ("保留率", format_ratio(counts.get("filtered"), counts.get("dedup")), "过滤后/过滤前"),
        ])

        show_logs("ref_filter", "参考CSV去重")
        preview_csv(st.session_state.outputs.get("filtered"), "参考去重结果")
        download_file(st.session_state.outputs.get("filtered"), "下载 filtered_main.csv")

        if config.get("update_reference"):
            st.markdown("**可选：覆盖更新reference.csv**")
            update_btn = st.button(
                "确认并覆盖 reference.csv",
                disabled=not st.session_state.step_done.get("dedup"),
                key="run_update_ref",
                width='stretch',
            )
            if update_btn:
                run_step(
                    "update_ref",
                    "覆盖更新reference.csv",
                    overwrite_reference_with_result,
                    str(st.session_state.outputs["dedup"]),
                    str(st.session_state.outputs["ref_path"]),
                    "utf-8-sig",
                    config.get("backup_reference", True),
                    True,
                )
            show_logs("update_ref", "覆盖更新reference.csv")
    else:
        st.markdown("<span class=\"chip chip-skip\">参考CSV去重 · 已跳过</span>", unsafe_allow_html=True)
        st.info("已关闭参考CSV去重，Step 3 自动跳过。")

with st.expander("Step 4 替换ptList", expanded=False):
    st.markdown(step_status_chip("replace_ptlist", "替换ptList"), unsafe_allow_html=True)
    replace_btn = st.button(
        "确认并执行 Step 4",
        disabled=not st.session_state.step_done.get("ref_filter"),
        key="run_replace",
        width='stretch',
    )
    if replace_btn:
        reset_downstream("replace_ptlist")
        processed_csv = output_root_path / "processed_replaced_ptlist.csv"
        excluded_csv = output_root_path / "processed_replaced_ptlist_excluded.csv"
        run_step(
            "replace_ptlist",
            "替换ptList",
            process_csv_replace_ptlist,
            str(st.session_state.outputs["filtered"]),
            str(processed_csv),
            str(excluded_csv),
        )
        st.session_state.outputs["processed"] = processed_csv
        st.session_state.outputs["processed_excluded"] = excluded_csv
        st.session_state.step_done["replace_ptlist"] = True
        counts["processed"] = get_row_count(processed_csv)
        counts["processed_excluded"] = get_row_count(excluded_csv)

    render_stats_cards([
        ("输入行数", format_int(counts.get("filtered")), "参考去重结果"),
        ("输出行数", format_int(counts.get("processed")), "ptList替换结果"),
        ("未筛选行数", format_int(counts.get("processed_excluded")), "未筛选数据"),
        ("保留率", format_ratio(counts.get("processed"), counts.get("filtered")), "输出/输入"),
    ])

    show_logs("replace_ptlist", "替换ptList")
    preview_csv(st.session_state.outputs.get("processed"), "ptList替换结果")
    download_file(st.session_state.outputs.get("processed"), "下载 processed_replaced_ptlist.csv")
    preview_csv(st.session_state.outputs.get("processed_excluded"), "未筛选数据（含原因）")
    download_file(st.session_state.outputs.get("processed_excluded"), "下载 processed_replaced_ptlist_excluded.csv")
    if st.session_state.outputs.get("processed_excluded"):
        st.markdown("**未筛选原因统计**")
        try:
            excluded_df = pd.read_csv(st.session_state.outputs.get("processed_excluded"), encoding="utf-8-sig")
            if "未筛选原因" in excluded_df.columns:
                reason_counts = (
                    excluded_df["未筛选原因"]
                    .fillna("未知")
                    .value_counts()
                    .reset_index()
                )
                reason_counts.columns = ["未筛选原因", "数量"]
                safe_dataframe(reason_counts, width='stretch')
            else:
                st.info("未筛选数据中未找到“未筛选原因”列。")
        except Exception as exc:
            st.warning(f"未筛选原因统计读取失败：{exc}")

with st.expander("Step 5 IoU筛选", expanded=False):
    st.markdown(step_status_chip("iou_filter", "IoU筛选"), unsafe_allow_html=True)
    iou_btn = st.button(
        "确认并执行 Step 5",
        disabled=not st.session_state.step_done.get("replace_ptlist"),
        key="run_iou",
        width='stretch',
    )
    if iou_btn:
        reset_downstream("iou_filter")
        high_iou_csv = output_root_path / f"high_iou_{config.get('iou_threshold', 0.98):.2f}.csv"
        other_csv = output_root_path / "other_data.csv"
        run_step(
            "iou_filter",
            "IoU筛选",
            filter_by_box_count_and_iou,
            str(st.session_state.outputs["processed"]),
            str(high_iou_csv),
            str(other_csv),
            int(config.get("min_boxes", 2)),
            float(config.get("iou_threshold", 0.98)),
        )
        st.session_state.outputs["high_iou"] = high_iou_csv
        st.session_state.outputs["other"] = other_csv
        st.session_state.step_done["iou_filter"] = True
        counts["high_iou"] = get_row_count(high_iou_csv)
        counts["other"] = get_row_count(other_csv)

    render_stats_cards([
        ("输入行数", format_int(counts.get("processed")), "ptList替换结果"),
        ("高IoU行数", format_int(counts.get("high_iou")), "满足阈值"),
        ("其他行数", format_int(counts.get("other")), "未满足阈值"),
        ("高IoU占比", format_ratio(counts.get("high_iou"), counts.get("processed")), "高IoU/输入"),
    ])

    show_logs("iou_filter", "IoU筛选")
    preview_csv(st.session_state.outputs.get("high_iou"), "高IoU结果")
    preview_csv(st.session_state.outputs.get("other"), "其他数据")
    download_file(st.session_state.outputs.get("high_iou"), "下载 high_iou.csv")
    download_file(st.session_state.outputs.get("other"), "下载 other_data.csv")

with st.expander("Step 5.5 标签替换", expanded=False):
    st.markdown(step_status_chip("label_replace", "标签替换"), unsafe_allow_html=True)
    label_map_path = st.session_state.outputs.get("label_map_path")
    label_map_ready = label_map_path is not None and Path(label_map_path).exists()
    if not label_map_ready:
        st.info("未上传标签对照表，步骤将自动跳过。")
    replace_label_btn = st.button(
        "确认并执行 Step 5.5",
        disabled=not st.session_state.step_done.get("iou_filter") or not label_map_ready,
        key="run_label_replace",
        width='stretch',
    )
    if replace_label_btn:
        reset_downstream("label_replace")
        replaced_csv = output_root_path / "other_data_label_replaced.csv"
        diff_path = output_root_path / "label_replace_diff.xlsx"
        unmatched_path = output_root_path / "label_replace_unmatched.xlsx"
        result = run_step(
            "label_replace",
            "标签替换",
            replace_labels_by_mapping,
            str(st.session_state.outputs.get("other")),
            str(label_map_path),
            str(replaced_csv),
            config.get("label_map_sheet"),
            config.get("label_map_old_col"),
            config.get("label_map_new_col"),
            None,
            str(diff_path),
            str(unmatched_path),
            30,
        )
        st.session_state.outputs["label_replaced"] = result.get("output_csv", replaced_csv)
        st.session_state.outputs["label_replace_summary"] = result.get("summary", {})
        st.session_state.outputs["label_replace_diff"] = result.get("diff", diff_path)
        st.session_state.outputs["label_replace_unmatched"] = result.get("unmatched", unmatched_path)
        st.session_state.outputs["label_replace_sample_diff"] = result.get("sample_diff", [])
        st.session_state.step_done["label_replace"] = True
        counts["label_replaced"] = get_row_count(replaced_csv)

    summary = st.session_state.outputs.get("label_replace_summary", {})
    if summary:
        render_stats_cards([
            ("映射数量", format_int(summary.get("mapping_size")), "对照表映射数"),
            ("替换行数", format_int(summary.get("replaced_rows")), "至少包含1个替换"),
            ("替换标签数", format_int(summary.get("replaced_labels")), "替换的标签总数"),
            ("无效JSON行数", format_int(summary.get("invalid_json_rows")), "标注字段解析失败"),
            ("未匹配标签数", format_int(summary.get("unmatched_labels")), "对照表未覆盖"),
        ])

    show_logs("label_replace", "标签替换")
    preview_csv(st.session_state.outputs.get("label_replaced"), "标签替换结果")
    download_file(st.session_state.outputs.get("label_replaced"), "下载 other_data_label_replaced.csv")
    if st.session_state.outputs.get("label_replace_unmatched"):
        st.markdown("**未匹配标签统计**")
        preview_csv(st.session_state.outputs.get("label_replace_unmatched"), "未匹配标签统计")
        download_file(st.session_state.outputs.get("label_replace_unmatched"), "下载 label_replace_unmatched.xlsx")
    if st.session_state.outputs.get("label_replace_diff"):
        st.markdown("**标签替换差异报告**")
        preview_csv(st.session_state.outputs.get("label_replace_diff"), "标签替换差异")
        download_file(st.session_state.outputs.get("label_replace_diff"), "下载 label_replace_diff.xlsx")
    sample_diff = st.session_state.outputs.get("label_replace_sample_diff") or []
    if sample_diff:
        st.markdown("**替换前后对比抽样**")
        safe_dataframe(pd.DataFrame(sample_diff), width='stretch')

with st.expander("Step 6 规则分类拆分", expanded=False):
    st.markdown(step_status_chip("split", "规则分类拆分"), unsafe_allow_html=True)
    rule_path = st.session_state.outputs.get("rule_path")
    rules_ready = rule_path is not None and Path(rule_path).exists()
    if config.get("rule_mode") == "两列映射" and (not config.get("rule_label_col") or not config.get("rule_category_col")):
        rules_ready = False
        st.warning("两列映射模式需要选择标签列和类别列。")
    if not rules_ready:
        st.warning("未找到分类规则文件，请在左侧选择并确认输入。")
    label_replace_done = st.session_state.step_done.get("label_replace")
    if not label_replace_done:
        st.warning("Step 6 将基于 Step 5.5 标签替换结果执行，请先完成 Step 5.5。")
    split_btn = st.button(
        "确认并执行 Step 6",
        disabled=not label_replace_done or not rules_ready,
        key="run_split",
        width='stretch',
    )
    if split_btn:
        reset_downstream("split")
        split_dir = output_root_path / "split_by_category"
        split_input = st.session_state.outputs.get("label_replaced")
        if not split_input:
            st.error("未找到 Step 5.5 的输出文件，请先执行标签替换。")
            st.stop()
        result = run_step(
            "split",
            "规则分类拆分",
            split_dataset_by_rules,
            str(split_input),
            str(rule_path),
            str(split_dir),
            "wide" if config.get("rule_mode") == "宽表(类别为列)" else "two_column",
            config.get("rule_sheet"),
            config.get("rule_label_col"),
            config.get("rule_category_col"),
            None,
            float(config.get("train_ratio", 0.8)),
            float(config.get("val_ratio", 0.1)),
            float(config.get("test_ratio", 0.1)),
            int(config.get("random_seed", 42)),
        )
        st.session_state.outputs["split_dir"] = split_dir
        st.session_state.outputs["category_files"] = result.get("category_files")
        st.session_state.outputs["unclassified"] = result.get("unclassified")
        st.session_state.outputs["split_counts"] = result.get("split_counts")
        st.session_state.outputs["classification_summary"] = result.get("summary", {})
        if st.session_state.outputs.get("unclassified"):
            summary_path = run_step(
                "unclassified_summary",
                "无法分类汇总",
                summarize_unclassified,
                str(st.session_state.outputs.get("unclassified")),
                str(split_dir),
                None,
            )
            st.session_state.outputs["unclassified_summary"] = summary_path
        st.session_state.step_done["split"] = True

    summary = st.session_state.outputs.get("classification_summary", {})
    render_stats_cards([
        ("可分类条数", format_int(summary.get("classified")), "多标签会拆分成多条"),
        ("无法分类条数", format_int(summary.get("unclassified")), "见 unclassified.xlsx"),
        ("类别数量", format_int(summary.get("categories")), "规则中匹配到的类别"),
    ])

    category_counts = summary.get("category_counts", {})
    if category_counts:
        st.markdown("**类别样本数统计**")
        count_df = pd.DataFrame(
            [{"类别": k, "样本数": v} for k, v in category_counts.items()]
        ).sort_values("样本数", ascending=False)
        safe_dataframe(count_df, width='stretch')
    else:
        st.info("暂无类别样本统计。")

    if st.session_state.outputs.get("category_files"):
        st.markdown("**分类Excel输出**")
        for path in st.session_state.outputs.get("category_files", []):
            st.write(f"`{path}`")
        st.markdown("**分类Excel条数统计**")
        stats_rows = []
        for path in st.session_state.outputs.get("category_files", []):
            try:
                xls = pd.ExcelFile(path)
                row_counts = {}
                for split in ["train", "val", "test"]:
                    if split in xls.sheet_names:
                        df_split = pd.read_excel(path, sheet_name=split)
                        row_counts[split] = len(df_split)
                    else:
                        row_counts[split] = 0
                total = row_counts["train"] + row_counts["val"] + row_counts["test"]
                stats_rows.append({
                    "类别": Path(path).stem,
                    "train": row_counts["train"],
                    "val": row_counts["val"],
                    "test": row_counts["test"],
                    "总计": total,
                })
            except Exception as exc:
                st.warning(f"读取分类Excel失败：{path}（{exc}）")
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows).sort_values("总计", ascending=False)
            safe_dataframe(stats_df, width='stretch')
    if st.session_state.outputs.get("unclassified"):
        download_file(st.session_state.outputs.get("unclassified"), "下载 unclassified.xlsx")

    if st.session_state.outputs.get("split_counts"):
        download_file(st.session_state.outputs.get("split_counts"), "下载 split_counts.xlsx")
        st.markdown("**拆分条数统计预览**")
        try:
            split_counts_df = pd.read_excel(st.session_state.outputs.get("split_counts"))
            safe_dataframe(split_counts_df, width='stretch')
            min_split = st.number_input("仅显示拆分条数 ≥ X 的图像", min_value=1, max_value=500, value=1, step=1, key="split_min_threshold")
            chart_df = split_counts_df[["source", "拆分条数"]].copy()
            chart_df["source"] = chart_df["source"].astype(str)
            chart_df = chart_df.sort_values("拆分条数", ascending=False)
            filtered_chart_df = chart_df[chart_df["拆分条数"] >= int(min_split)]
            if filtered_chart_df.empty:
                st.info("没有满足条件的图像。")

            st.markdown("**拆分最多的 TOP N 图像**")
            top_n = st.number_input("TOP N", min_value=5, max_value=50, value=10, step=5, key="split_top_n")
            top_df = filtered_chart_df.head(int(top_n)) if not filtered_chart_df.empty else chart_df.head(int(top_n))
            safe_dataframe(top_df, width='stretch')
        except Exception as exc:
            st.warning(f"split_counts 读取失败：{exc}")

    if st.session_state.outputs.get("unclassified_summary"):
        download_file(st.session_state.outputs.get("unclassified_summary"), "下载 unclassified_summary.xlsx")
        sheet_choice = st.selectbox(
            "选择汇总表",
            options=["label_summary", "reason_summary", "reason_label"],
            index=0,
            key="unclassified_sheet_choice",
        )
        try:
            summary_path = st.session_state.outputs.get("unclassified_summary")
            summary_df = pd.read_excel(summary_path, sheet_name=sheet_choice)
            download_dataframe_excel(
                summary_df,
                f"{sheet_choice}.xlsx",
                f"下载 {sheet_choice}.xlsx",
                key=f"download_{sheet_choice}",
            )
            st.caption(f"当前表：{len(summary_df)} 行 · {len(summary_df.columns)} 列")
        except Exception as exc:
            st.warning(f"汇总表读取失败：{exc}")
        st.markdown("**无法分类汇总预览**")
        try:
            safe_dataframe(summary_df, width='stretch')
        except Exception as exc:
            st.warning(f"{sheet_choice} 读取失败：{exc}")

with st.expander("Step 7 生成YOLO数据集", expanded=False):
    st.markdown(step_status_chip("yolo", "生成YOLO数据集"), unsafe_allow_html=True)
    yolo_ready = st.session_state.step_done.get("split")
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    progress_text.caption("等待开始…")
    yolo_resume = st.checkbox(
        "断点续存（跳过已有图片与标签）",
        value=True,
        key="yolo_resume",
    )
    yolo_btn = st.button(
        "确认并执行 Step 7",
        disabled=not yolo_ready,
        key="run_yolo",
        width='stretch',
    )
    if yolo_btn:
        reset_downstream("yolo")
        yolo_dir = output_root_path / "yolo_datasets"
        def _progress_cb(done, total, downloaded, category=None, split=None, filename=None, label=None, excel=None, row_idx=None):
            if total and total > 0:
                progress_bar.progress(min(done / total, 1.0))
                extra = ""
                if category:
                    extra += f" | 当前类别：{category}"
                if split:
                    extra += f" | 当前split：{split}"
                if filename:
                    extra += f" | 当前文件：{filename}"
                if label:
                    extra += f" | 当前标签：{label}"
                if excel:
                    extra += f" | 当前Excel：{excel}"
                if row_idx is not None:
                    extra += f" | 当前行：{row_idx}"
                progress_text.markdown(f"已处理 {done}/{total} 条，已下载 {downloaded} 张{extra}")
            else:
                progress_bar.progress(0.0)
                progress_text.markdown("未找到可处理的数据")
        result = run_step(
            "yolo",
            "生成YOLO数据集",
            generate_yolo_datasets_from_excels,
            st.session_state.outputs.get("category_files", []),
            str(yolo_dir),
            str(yolo_dir / "image_cache"),
            "source",
            "分类标签",
            "新_结果字段-目标检测标签配置",
            "结果字段-目标检测标签配置",
            "width",
            "height",
            True,
            int(config.get("random_seed", 42)),
            config.get("class_order") or None,
            yolo_resume,
            _progress_cb,
        )
        st.session_state.outputs["yolo_dir"] = yolo_dir
        st.session_state.outputs["yolo_datasets"] = result.get("datasets")
        st.session_state.outputs["yolo_skipped"] = result.get("skipped")
        st.session_state.outputs["yolo_stats"] = result.get("stats", {})
        st.session_state.outputs["yolo_progress"] = {
            "total": result.get("total"),
            "processed": result.get("processed"),
            "downloaded": result.get("downloaded"),
        }
        st.session_state.outputs["yolo_dataset_name_map"] = result.get("dataset_name_map", {})
        yolo_label_stats, yolo_label_df = summarize_yolo_label_counts(result.get("datasets"))
        st.session_state.outputs["yolo_label_stats"] = yolo_label_stats
        st.session_state.outputs["yolo_label_df"] = yolo_label_df
        st.session_state.step_done["yolo"] = True

    if st.session_state.outputs.get("yolo_dir"):
        st.write(f"YOLO数据集输出目录：`{st.session_state.outputs.get('yolo_dir')}`")
    if st.session_state.outputs.get("yolo_datasets"):
        st.markdown("**已生成的数据集**")
        for path in st.session_state.outputs.get("yolo_datasets", []):
            st.write(f"`{path}`")
    yolo_progress = st.session_state.outputs.get("yolo_progress")
    if yolo_progress:
        render_stats_cards([
            ("总条数", format_int(yolo_progress.get("total")), "待处理数据"),
            ("已处理", format_int(yolo_progress.get("processed")), "已完成转换"),
            ("已下载", format_int(yolo_progress.get("downloaded")), "图像成功写入"),
        ])
    yolo_stats = st.session_state.outputs.get("yolo_stats", {})
    if yolo_stats:
        st.markdown("**各类别拆分统计**")
        stats_rows = []
        for category, splits in yolo_stats.items():
            stats_rows.append({
                "类别": category,
                "train": splits.get("train", 0),
                "val": splits.get("val", 0),
                "test": splits.get("test", 0),
                "总计": splits.get("train", 0) + splits.get("val", 0) + splits.get("test", 0),
            })
        stats_df = pd.DataFrame(stats_rows).sort_values("总计", ascending=False)
        safe_dataframe(stats_df, width='stretch')
    yolo_label_stats = st.session_state.outputs.get("yolo_label_stats", {})
    yolo_label_df = st.session_state.outputs.get("yolo_label_df")
    yolo_dataset_name_map = st.session_state.outputs.get("yolo_dataset_name_map", {})
    if yolo_label_stats:
        st.markdown("**YOLO数据集标签统计（按图片数）**")
        if yolo_label_df is not None and not yolo_label_df.empty:
            st.markdown("**导出标签统计**")
            download_dataframe_excel(
                yolo_label_df,
                f"yolo_label_stats_{st.session_state.run_id}.xlsx",
                "下载 标签统计 Excel",
                key="download_yolo_label_excel",
            )
            csv_bytes = yolo_label_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="下载 标签统计 CSV",
                data=csv_bytes.encode("utf-8-sig"),
                file_name=f"yolo_label_stats_{st.session_state.run_id}.csv",
                mime="text/csv",
            )
        for dataset_name, split_stats in yolo_label_stats.items():
            display_name = yolo_dataset_name_map.get(dataset_name) or dataset_name
            title = f"{display_name} 标签统计"
            if display_name != dataset_name:
                title = f"{display_name}（{dataset_name}） 标签统计"
            with st.expander(title, expanded=False):
                for split in ["train", "val", "test", "all"]:
                    split_info = split_stats.get(split, {})
                    total_images = split_info.get("total_images", 0)
                    img_counts = split_info.get("label_counts", {})
                    box_counts = split_info.get("box_counts", {})
                    split_label = "汇总" if split == "all" else split
                    st.markdown(f"**{split_label} 标签统计**")
                    st.caption(f"split总图片数：{total_images}")
                    if img_counts or box_counts:
                        rows = []
                        all_labels = set(img_counts) | set(box_counts)
                        for label in all_labels:
                            img_count = img_counts.get(label, 0)
                            box_count = box_counts.get(label, 0)
                            ratio = (img_count / total_images) if total_images else 0.0
                            rows.append({
                                "标签": label,
                                "图片数量": img_count,
                                "标注框数量": box_count,
                                "占比": f"{ratio * 100:.1f}%",
                            })
                        df = pd.DataFrame(rows).sort_values("图片数量", ascending=False)
                        safe_dataframe(df, width='stretch')
                    else:
                        st.info("暂无标签数据。")
    if st.session_state.outputs.get("yolo_skipped"):
        download_file(st.session_state.outputs.get("yolo_skipped"), "下载 yolo_skipped.xlsx")

with st.expander("Step 8 下载并绘制标注图片", expanded=False):
    if config.get("run_download"):
        st.markdown(step_status_chip("download", "下载并绘制标注图片"), unsafe_allow_html=True)
        download_btn = st.button(
            "确认并执行 Step 8",
            disabled=not st.session_state.step_done.get("yolo"),
            key="run_download",
            width='stretch',
        )
        if download_btn:
            draw_input = st.session_state.outputs.get("label_replaced") or st.session_state.outputs.get("other")
            run_step(
                "download",
                "下载并绘制标注图片",
                download_and_draw_annotations,
                str(draw_input),
                str(output_root_path),
                None,
                None,
                config.get("max_images"),
                15,
            )
            st.session_state.outputs["download_dir"] = output_root_path / "downloaded_images"
            st.session_state.outputs["annotated_dir"] = output_root_path / "annotated_images"
            st.session_state.step_done["download"] = True

        image_count = get_image_count(st.session_state.outputs.get("annotated_dir"))
        render_stats_cards([
            ("输出图片数", format_int(image_count), "标注图片"),
            ("限制数量", format_int(config.get("max_images") or 0), "0表示不限"),
        ])

        show_logs("download", "下载并绘制标注图片")

        annotated_dir = st.session_state.outputs.get("annotated_dir")
        if annotated_dir and annotated_dir.exists():
            images = list(annotated_dir.glob("*"))[:12]
            if images:
                st.image([str(p) for p in images], caption=[p.name for p in images])
            else:
                st.info("暂无标注图片可预览。")
    else:
        st.markdown("<span class=\"chip chip-skip\">图片标注 · 已跳过</span>", unsafe_allow_html=True)
        st.info("已关闭图片标注步骤，Step 8 自动跳过。")

st.markdown("---")

st.markdown("**结果导出区**")
export_left, export_right = st.columns([2, 1])
with export_left:
    only_classification = st.checkbox("只打包分类结果", value=False)
    include_yolo = st.checkbox("打包YOLO数据集", value=False)
    include_images = st.checkbox(
        "打包包含图片（下载原图与标注图）",
        value=False,
        disabled=not config.get("run_download") or only_classification,
    )
    if only_classification:
        st.info("当前仅导出分类结果（类别Excel + unclassified）。")
    elif include_images and not config.get("run_download"):
        st.info("未启用图片标注步骤，无法包含图片。")
with export_right:
    zip_buffer = build_export_zip(
        st.session_state.outputs,
        include_images=include_images,
        only_classification=only_classification,
    )
    zip_name = "classification_only" if only_classification else "yolo_pipeline"
    st.download_button(
        label="下载全部结果 ZIP",
        data=zip_buffer,
        file_name=f"{zip_name}_{st.session_state.run_id}.zip",
        mime="application/zip",
        width='stretch',
    )
    if include_yolo:
        yolo_zip = build_yolo_zip(st.session_state.outputs.get("yolo_dir"))
        if yolo_zip:
            st.download_button(
                label="下载 YOLO 数据集 ZIP",
                data=yolo_zip,
                file_name=f"yolo_dataset_{st.session_state.run_id}.zip",
                mime="application/zip",
                width='stretch',
            )
        else:
            st.info("尚未生成 YOLO 数据集。")

st.markdown("---")
st.markdown("**流程日志汇总**")
log_steps = [
    ("merge", "合并CSV"),
    ("dedup", "按source去重"),
    ("ref_filter", "参考CSV去重"),
    ("update_ref", "覆盖reference.csv"),
    ("replace_ptlist", "替换ptList"),
    ("iou_filter", "IoU筛选"),
    ("split", "规则分类拆分"),
    ("unclassified_summary", "无法分类汇总"),
    ("yolo", "生成YOLO数据集"),
    ("download", "下载并绘制标注图片"),
]
visible_tabs = []
visible_labels = []
for key, label in log_steps:
    if key == "ref_filter" and not config.get("use_reference"):
        continue
    if key == "update_ref" and not config.get("update_reference"):
        continue
    if key == "download" and not config.get("run_download"):
        continue
    visible_tabs.append(key)
    visible_labels.append(label)

if visible_tabs:
    tabs = st.tabs(visible_labels)
    for idx, key in enumerate(visible_tabs):
        with tabs[idx]:
            log_text = st.session_state.logs.get(key)
            if log_text:
                st.text_area("日志输出", log_text, height=220)
            else:
                st.info("暂无日志。")
