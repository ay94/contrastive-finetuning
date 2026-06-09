import streamlit as st
from pathlib import Path
from Utils import ExperimentController
from contrastive_experiment.utils import get_base_folder
@st.cache_data
def load_all_data(base_folder):
    controller = ExperimentController(base_folder)
    model_dirs = controller.get_model_dirs()
    
    combined_metrics = controller.get_model().get_tab_data(model_dirs, "combined_metrics")
    reports = controller.get_model().get_tab_data(model_dirs, "reports")
    average_similarities = controller.get_model().get_tab_data(model_dirs, "average_similarities")
    similarity_matrices = controller.get_model().get_tab_data(model_dirs, "similarity_matrices")
    semantic_maps = controller.get_model().get_tab_data(model_dirs, "semantic_maps")
    unmasked_annotated_datasets = controller.get_model().get_tab_data(model_dirs, "unmasked_annotated_datasets")
    
    return {
        "combined_metrics": combined_metrics,
        "reports": reports,
        "average_similarities": average_similarities,
        "similarity_matrices": similarity_matrices,
        "semantic_maps": semantic_maps,
        "unmasked_annotated_datasets": unmasked_annotated_datasets
    }


def main():
    st.header("Contrastive Experiments Dashboard")
    

    _pages_dir = Path(__file__).parent / "pages"
    pages = {
        "Choose Model": str(_pages_dir / "1_Choose_Model.py"),
        "Classification Report": str(_pages_dir / "2_Classification_Report.py"),
        "Average Similarity": str(_pages_dir / "3_Average_Similarity.py"),
        "Similarity Matrix": str(_pages_dir / "4_Similarity_Matrix.py"),
        "Semantic Map": str(_pages_dir / "5_Semantic_Map.py"),
        "Class Metrics": str(_pages_dir / "6_Class_Metrics.py"),
        "Unmasked Data": str(_pages_dir / "7_Unmasked_Data.py"),
    }
    st.sidebar.title("Experiment Evaluation Stages")
    page = st.sidebar.selectbox("Select Page", options=list(pages.keys()))
    base_folder = get_base_folder()
    st.session_state["base_folder"] = base_folder
    
    if "all_data" not in st.session_state:
        with st.spinner('Loading data...'):
            # Load and cache all data
            all_data = load_all_data(base_folder)
            st.session_state["all_data"] = all_data
    
    
    
    if page in pages:
        # st.experimental_set_query_params(page=page)
        with open(pages[page]) as f:
            code = compile(f.read(), pages[page], 'exec')
            exec(code, globals())

if __name__ == "__main__":
    main()
