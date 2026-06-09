import streamlit as st

def class_metrics_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Class Label Metrics"
    st.header(tab_key)
    
    all_data = st.session_state["all_data"]
    tab_data = all_data["reports"]

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
    filters["Split"] = "test"

    if st.button("Generate Class Metrics", key=f"{tab_key}_class_metrics"):
        with st.spinner('Processing data...'):
            report = controller.get_model().get_filtered_clean_data(model_dirs, "reports", filters)
            average_similarity = controller.get_model().get_filtered_clean_data(model_dirs, "average_similarities", filters)
            semantic_map = (
                controller.get_model().get_filtered_clean_data(model_dirs, "semantic_maps", filters)
                .groupby("class_label")["true_score"]
                .mean()
                .reset_index()
            )
            annotated_data = (
                controller.get_model().get_filtered_clean_data(model_dirs, "annotated_datasets", filters)
                .groupby("class_label")["pred_score"]
                .mean()
                .reset_index()
            )

            class_metrics = controller.get_model().aggregate_data(
                [report, average_similarity, semantic_map, annotated_data]
            )

            view.display_tab_data(class_metrics)

            melted_data = controller.get_model().melt_data(class_metrics)
            fig = controller.get_model().create_scatter_plot(
                melted_data,
                x="class_label",
                y="Value",
                color="Metric",
                title="Different Metrics Per Class",
                opacity=0.7,
                marker_size=5,
            )
            view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    class_metrics_page(controller)
else:
    st.error("Please run the main app script.")
