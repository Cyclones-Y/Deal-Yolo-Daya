import io
import streamlit as st
import pandas as pd
from pathlib import Path
from ..core.utils import format_duration, format_bytes

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

def render_icon_tree_custom(nodes, level=0):
    if not nodes:
        return ""
    html = ""
    for node in nodes:
        indent = level * 20
        icon_html = """<span class="fm-icon"><svg viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"></path></svg></span>"""
        
        node_class = f"fm-depth-{min(level, 4)}"
        
        html += f"""
        <div class="fm-node {node_class}" style="margin-left:{indent}px;">
            {icon_html}
            <span class="fm-name">{node['label']}</span>
            <span style="flex:1;"></span>
            <span class="fm-path" title="{node['value']}">{node['value']}</span>
        </div>
        """
        if "children" in node and node["children"]:
            html += render_icon_tree_custom(node["children"], level + 1)
    return html

def render_advanced_tree_component(root_path: Path):
    if not root_path.exists():
        st.warning("目录不存在")
        return
    
    # 简单的递归构建树结构
    def build_tree(path, depth=0, max_depth=3):
        if depth > max_depth:
            return []
        nodes = []
        try:
            for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if p.name.startswith("."): continue
                node = {
                    "label": p.name,
                    "value": str(p),
                    "children": []
                }
                if p.is_dir():
                    node["children"] = build_tree(p, depth + 1, max_depth)
                nodes.append(node)
        except Exception:
            pass
        return nodes

    tree_data = build_tree(root_path)
    # 使用自定义HTML渲染树
    html_content = f"""
    <div class="file-manager">
        {render_icon_tree_custom(tree_data)}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def render_copy_button(text_to_copy: str, label: str = "复制", key: str = None):
    # Streamlit 原生暂无复制到剪贴板功能，通常使用 code 块
    st.code(text_to_copy, language="text")

def show_confirm_dialog(key_confirm, title, message, on_confirm):
    if st.session_state.get(key_confirm):
        st.markdown(
            f"""
            <div style="background:#fff1f2;border:1px solid #fda4af;padding:16px;border-radius:12px;margin-bottom:16px;">
                <h4 style="color:#be123c;margin-top:0;">{title}</h4>
                <p style="color:#881337;">{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("确定执行", key=f"{key_confirm}_yes", type="primary"):
                on_confirm()
                st.session_state[key_confirm] = False
                st.rerun()
        with col2:
            if st.button("取消", key=f"{key_confirm}_no"):
                st.session_state[key_confirm] = False
                st.rerun()

def render_run_visualization(run_dir: Path):
    if not run_dir.exists():
        st.warning("未找到训练结果目录")
        return

    st.markdown(f"### 训练结果: {run_dir.name}")
    
    # 显示结果图片
    image_files = sorted(list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg")))
    
    tabs = st.tabs(["关键指标", "混淆矩阵", "PR曲线", "预测样例", "训练日志"])
    
    with tabs[0]:
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                df.columns = [c.strip() for c in df.columns]
                st.dataframe(df, use_container_width=True)
                
                # 绘制简单的损失曲线
                cols_to_plot = [c for c in df.columns if "loss" in c.lower() or "map" in c.lower()]
                if cols_to_plot:
                    st.line_chart(df[cols_to_plot])
            except Exception as e:
                st.error(f"读取results.csv失败: {e}")
        
        # 显示 results.png 如果存在
        results_img = run_dir / "results.png"
        if results_img.exists():
            st.image(str(results_img), caption="Results Summary")

    with tabs[1]:
        cm_img = run_dir / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix")
        
        cm_norm_img = run_dir / "confusion_matrix_normalized.png"
        if cm_norm_img.exists():
            st.image(str(cm_norm_img), caption="Normalized Confusion Matrix")

    with tabs[2]:
        pr_curve = run_dir / "PR_curve.png"
        if pr_curve.exists():
            st.image(str(pr_curve), caption="PR Curve")
        
        f1_curve = run_dir / "F1_curve.png"
        if f1_curve.exists():
            st.image(str(f1_curve), caption="F1 Curve")

    with tabs[3]:
        # 显示验证集预测图片
        val_imgs = sorted(list((run_dir).glob("val_batch*_pred.jpg")))
        if val_imgs:
            st.image([str(p) for p in val_imgs[:4]], caption=[p.name for p in val_imgs[:4]], width=300)
        else:
            st.info("未找到验证集预测图片")

    with tabs[4]:
        # 显示 args.yaml
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            with st.expander("训练配置 (args.yaml)"):
                st.code(args_yaml.read_text(encoding="utf-8"), language="yaml")
