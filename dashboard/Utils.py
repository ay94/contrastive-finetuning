
import json
import logging
import re
from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report

from contrastive_experiment.utils import get_base_folder
from contrastive_experiment.results_utils import DataManager

class ExperimentData:
    def __init__(self, base_folder):
        self.base_folder = base_folder.expanduser()
        self.data_cache = {}  # Initialize cache dictionary

    def get_models(self):
        base_dir = self.base_folder / "contrastive-experiment"
        if not base_dir.exists():
            return {}
        return {dir.name: dir for dir in base_dir.iterdir() if dir.is_dir()}

    @st.cache_data
    def get_tab_data(_self, models, analysis_type):
        combined_metrics = []
        for model_dir in models.values():
            manager = DataManager(model_dir) 
            df = manager.load_data(analysis_type)
            combined_metrics.append(df)
        return pd.concat(combined_metrics)


    def filter_tab_data(self, df, filters):
        for key, value in filters.items():
            if value != "All" and key in df.columns:
                df = df[df[key] == value]
        return df
    
    
    def create_similarity_matrix(self, df):
        fig = px.imshow(df, color_continuous_scale="Viridis", title="Similarity Matrix")
        fig.update_layout(
            coloraxis_colorbar=dict(title="Similarity Score"),
            autosize=False,
            width=800,
            height=800,
        )
        fig.update_traces(
            hoverinfo="x+y+z",
            hovertemplate="Var X: %{x}<br>Var Y: %{y}<br>Intensity: %{z}",
        )
        return fig
    
    def create_scatter_plot(self, df, x, y, color, title, opacity, marker_size):
        fig = px.scatter(df, x=x, y=y, color=color, opacity=opacity, title=title)
        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(
            xaxis_tickangle=90,  
            xaxis_title="Index",
            yaxis_title="Value",
            autosize=True,
            width=1200,  
            height=700,  
            hovermode="closest",
        )
        return fig

    def get_filtered_clean_data(self, models, data_key, filters):
        tab_data = self.get_tab_data(models, data_key)
        for key, value in filters.items():
            if value != "All" and key in tab_data.columns:
                tab_data = tab_data[tab_data[key] == value]
        tab_data = self.clean_data(tab_data)
        return tab_data

    def clean_data(self, df):
        df.rename(
            columns={"class": "class_label", "Class Label": "class_label"}, inplace=True
        )
        columns_to_remove = [
            "Experiment",
            "Model Name",
            "Config",
            "Baseline",
            "Model",
            "Split",
            "support",
        ]
        df = df.drop(
            columns=[
                col
                for col in df.columns
                if any(sub in col for sub in columns_to_remove)
            ]
        )
        rows_to_remove = ["accuracy", "macro avg", "weighted avg"]
        df = df[~df["class_label"].isin(rows_to_remove)]
        return df

    def melt_data(self, df):
        melted_df = df.melt(
            id_vars=["class_label"], var_name="Metric", value_name="Value"
        )
        return melted_df

    def aggregate_data(self, dataframes):
        return reduce(
            lambda left, right: pd.merge(left, right, on="class_label", how="outer"),
            dataframes,
        )


class DashboardView:
    def display_tab_data(self, tab_data):
        st.write(tab_data)

    def select_model(self, model_names, key):
        selected_model = st.selectbox("Select Model Name", model_names, key=key)
        st.session_state["selected_model_name"] = (
            selected_model  
        )
        return selected_model

    def header(self, text):
        st.header(text)

    def display_model_name(self, model_name):
        st.write(f"Selected Model Name: {model_name}")

    def select_column(self, label, options, key):
        return st.selectbox(label, options, key=key)

    def display_plot(self, plot):
        st.plotly_chart(plot)

    def get_stored_model_name(self):
        # Check if the key exists in the session state to avoid KeyError
        return st.session_state.get("selected_model_name", default=None)

    def remove_none_option(self, options):
        # Filter out None or NA values and remove 'All' if included
        return [option for option in options if option]

    def display_message(self, message):
        st.write(message)

    def create_selection_boxes(self, df, filter_columns, data_key):
        filters = {}
        for column in filter_columns:
            options = sorted(df[column].dropna().unique())
            selected_value = st.selectbox(
                f"Select {column}",
                options=self.remove_none_option(options) + ["All"],
                key=f"{data_key}_{column}_filter",
            )
            filters[column] = selected_value
        return filters

    def get_masking_preference(self):
        return st.checkbox(
            "Apply masking to the data", value=False, key="masking_check"
        )


class ExperimentController:
    def __init__(self, base_folder):
        self.model = ExperimentData(base_folder)
        self.view = DashboardView()
        self.model_dirs = self.model.get_models()
        
    def get_model(self):
        return self.model
    
    def get_view(self):
        return self.view
    
    def get_model_dirs(self):
        return self.model_dirs
    