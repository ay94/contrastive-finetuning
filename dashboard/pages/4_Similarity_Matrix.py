import streamlit as st

def similarity_matrix_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Similarity Confusion Matrix"
    st.header(tab_key)
    all_data = st.session_state["all_data"]
    tab_data = all_data["similarity_matrices"]
    
    # tab_data = controller.get_model().get_tab_data(model_dirs, "similarity_matrices")
    
    model_selection = view.select_column(
        "Select Model", list(tab_data["Model"].unique()), key=f"{tab_key}_model"
    )
    
    filters = {"Model": model_selection}
    
    if model_selection != "Baseline":
        model_data = tab_data[tab_data['Model'] == model_selection]
        config_selection = view.select_column(
            "Select Config",
            view.remove_none_option(list(model_data["Config"].unique())),
            key=f"{tab_key}_config",
        )
        filters["Config"] = config_selection

    split_selection = view.select_column(
        "Select Split", list(tab_data["Split"].unique()), key=f"{tab_key}_split"
    )
    filters["Split"] = split_selection
    filtered_data = controller.get_model().filter_tab_data(tab_data, filters)
    filtered_data = filtered_data.iloc[:, :-5]
    filtered_data = filtered_data.set_index("index")
    filtered_data.index.name = None
    view.display_tab_data(filtered_data)
    st.write(f"Number of classes: {len(filtered_data)}")
    if st.button("Generate Similarity Matrix", key=f"{tab_key}_similarity_matrix"):
        fig = controller.get_model().create_similarity_matrix(filtered_data)
        view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    similarity_matrix_page(controller)
else:
    st.error("Please run the main app script.")
