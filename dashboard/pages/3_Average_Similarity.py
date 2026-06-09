import streamlit as st

def average_similarity_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Average Similarity Per Class"
    st.header(tab_key)
    all_data = st.session_state["all_data"]
    tab_data = all_data["average_similarities"]
    selected_model = view.get_stored_model_name()
    view.display_model_name(selected_model)
    # tab_data = controller.get_model().get_tab_data(model_dirs, "average_similarities")
    view.display_tab_data(tab_data)
    color_selection = view.select_column("Select Average Color", tab_data.columns[2:], key=f"{tab_key}_color_selection")
    fig = controller.get_model().create_scatter_plot(
        tab_data,
        x="Class Label",
        y="Average Similarity",
        color=color_selection,
        title="Average Similarity Per Class Label",
        opacity=0.7,
        marker_size=5,
    )
    view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    average_similarity_page(controller)
else:
    st.error("Please run the main app script.")
