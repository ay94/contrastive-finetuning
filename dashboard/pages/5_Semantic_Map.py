import streamlit as st

def semantic_map_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Semantic Map"
    st.header(tab_key)
    plot_config = {
        "x": "x",
        "y": "y",
        "color": "class_label",
        "title": "Semantic Map",
        "opacity": 0.7,
        "marker_size": 2,
    }
    all_data = st.session_state["all_data"]
    tab_data = all_data["semantic_maps"]
    
    # tab_data = controller.get_model().get_tab_data(model_dirs, "semantic_maps")

    filters = view.create_selection_boxes(tab_data, ["Model", "Split"], tab_key)
    if filters["Model"] != "Baseline":
        model_data = tab_data[tab_data['Model'] == filters["Model"]]
        config_selection = view.select_column(
            "Select Config",
            view.remove_none_option(list(model_data["Config"].unique())),
            key=f"{tab_key}_config",
        )
        filters["Config"] = config_selection

    filtered_data = controller.get_model().filter_tab_data(tab_data, filters)
    view.display_tab_data(filtered_data)

    if st.button("Generate Semantic Map", key=f"{tab_key}_semantic_map"):
        fig = controller.get_model().create_scatter_plot(
            filtered_data,
            x=plot_config["x"],
            y=plot_config["y"],
            color=plot_config["color"],
            title=plot_config["title"],
            opacity=plot_config["opacity"],
            marker_size=plot_config["marker_size"],
        )
        view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    semantic_map_page(controller)
else:
    st.error("Please run the main app script.")
