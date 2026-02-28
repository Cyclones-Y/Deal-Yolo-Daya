import sys
from pathlib import Path
import streamlit as st
from datetime import datetime

# Add src to sys.path to ensure imports work correctly if run from different locations
# This helps when running `streamlit run src/deal_yolo_data/app.py`
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.deal_yolo_data.ui.styles import inject_style
from src.deal_yolo_data.ui.pages.processing import render_processing_pipeline
from src.deal_yolo_data.ui.pages.training import render_training_platform

def init_session_state():
    if "run_id" not in st.session_state:
        st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "step_done" not in st.session_state:
        st.session_state.step_done = {}
    if "logs" not in st.session_state:
        st.session_state.logs = {}
    if "outputs" not in st.session_state:
        st.session_state.outputs = {}
    if "config" not in st.session_state:
        st.session_state.config = {
            "use_reference": True,
            "update_reference": False,
            "backup_reference": True,
            "merge_chunk_size": 100000,
            "keep_outputs": True,
            "min_boxes": 2,
            "iou_threshold": 0.98,
            "run_download": False,
            "max_images": None,
            "ref_mode": "上传参考CSV",
            "rule_mode": "宽表(类别为列)",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": 42,
        }
    if "input_ready" not in st.session_state:
        st.session_state.input_ready = False
    if "output_root" not in st.session_state:
        st.session_state.output_root = str(Path.cwd() / "runs" / st.session_state.run_id)

def main():
    st.set_page_config(
        page_title="YOLO Data & Training Platform",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()
    init_session_state()

    st.sidebar.title("🚀 导航")
    page = st.sidebar.radio("选择功能模块", ["数据处理流水线", "YOLO 训练平台"])

    if page == "数据处理流水线":
        render_processing_pipeline()
    elif page == "YOLO 训练平台":
        render_training_platform()

if __name__ == "__main__":
    main()
