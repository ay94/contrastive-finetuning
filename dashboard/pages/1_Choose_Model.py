import streamlit as st
from Utils import ExperimentController
from pathlib import Path

def choose_model_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Model Selection"
    st.header(tab_key)
    all_data = st.session_state["all_data"]
    tab_data = all_data["combined_metrics"]
    selected_model = view.select_model(list(model_dirs.keys()), key=f"{tab_key}_model")
    # tab_data = controller.get_model().get_tab_data(model_dirs, "combined_metrics")
    view.display_model_name(selected_model)
    sort_by = view.select_column("Sort By", tab_data.columns[1:-4], key=f"{tab_key}_sorted_by")
    view.display_tab_data(tab_data.sort_values(by=sort_by, ascending=False))
    y_selection = view.select_column("Select Y", tab_data.columns[1:-4], key=f"{tab_key}_y_selection")
    fig = controller.get_model().create_scatter_plot(
        tab_data,
        x="Model",
        y=y_selection,
        color='Config',
        title="Classification Report",
        opacity=0.7,
        marker_size=5,
    )
    view.display_plot(fig)


if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    choose_model_page(controller)
else:
    st.error("Please run the main app script.")
