import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report

def unmasked_data_page(controller):
    view = controller.get_view()
    model_dirs = controller.get_model_dirs()
    
    tab_key = "Unmasked Data"
    st.header(tab_key)
    all_data = st.session_state["all_data"]
    tab_data = all_data["unmasked_annotated_datasets"]

    # tab_data = controller.get_model().get_tab_data(model_dirs, "unmasked_annotated_datasets")
    if view.get_masking_preference():
        mask = (tab_data["topicId"] != "topic_111") & ~(
            tab_data["class_label"].isin(["class_111", "class_-01"])
        )
        tab_data = tab_data[mask]

    model_selection = view.select_column(
        "Select Model", list(tab_data["Model"].unique()), key=f"{tab_key}_model"
    )
    filters = {"Model": model_selection}

    if model_selection != "Baseline":
        config_selection = view.select_column(
            "Select Config",
            view.remove_none_option(list(tab_data["Config"].unique())),
            key=f"{tab_key}_config",
        )
        filters["Config"] = config_selection
    filtered_data = controller.get_model().filter_tab_data(tab_data, filters)

    view.display_tab_data(filtered_data)

    group_by_column = view.select_column(
        "Groupby", ["labeled_by", "type", "source"], key=f"{tab_key}_groupby"
    )
    grouped = filtered_data.groupby(group_by_column)
    reports = []
    for name, group in grouped:
        report = classification_report(
            group["class_label"], group["predicted_label"], output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df[group_by_column] = name
        reports.append(report_df)
    aggregated_reports = pd.concat(reports)

    if st.button("Generate Report", key=f"{tab_key}_classification_report"):
        y_selection = view.select_column(
            "Select Y Metric", report_df.columns, key=f"{tab_key}_y_selection"
        )

        fig = controller.get_model().create_scatter_plot(
            aggregated_reports.reset_index(),
            x="index",
            y=y_selection,
            color=group_by_column,
            title=f"Classification Report aggregated by {group_by_column}",
            opacity=0.7,
            marker_size=5,
        )
        view.display_plot(fig)

if "base_folder" in st.session_state:
    from Utils import ExperimentController
    controller = ExperimentController(st.session_state["base_folder"])
    unmasked_data_page(controller)
else:
    st.error("Please run the main app script.")
