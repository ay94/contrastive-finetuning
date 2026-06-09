import streamlit as st

def classification_report_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Classification Reports"
    st.header(tab_key)
    all_data = st.session_state["all_data"]
    tab_data = all_data["reports"]
    # tab_data = controller.get_model().get_tab_data(model_dirs, "reports")
    selected_model = view.get_stored_model_name()
    view.display_model_name(selected_model)
    
    y_selection = view.select_column("Select Y", tab_data.columns[1:-4], key=f"{tab_key}_y_selection")
    color_selection = view.select_column("Select Color", tab_data.columns[-2:], key=f"{tab_key}_color_selection")
    fig = controller.get_model().create_scatter_plot(
        tab_data,
        x="class",
        y=y_selection,
        color=color_selection,
        title="Classification Report",
        opacity=0.7,
        marker_size=5,
    )
    view.display_plot(fig)
    filters = view.create_selection_boxes(tab_data, ["Config", "Model"], tab_key)
    filtered_data = controller.get_model().filter_tab_data(tab_data, filters)
    fig = controller.get_model().create_scatter_plot(
        filtered_data,
        x="class",
        y=y_selection,
        color=color_selection,
        title="Classification Report",
        opacity=0.7,
        marker_size=5,
    )
    view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    classification_report_page(controller)
else:
    st.error("Please run the main app script.")
