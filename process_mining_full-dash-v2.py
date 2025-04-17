import streamlit as st
import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.filtering.log.variants import variants_filter
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Functions from process_mining_full-v3.py (excluding social network parts)
# ----------------------------------------------------------------------------------
def load_and_prepare_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df_melted = pd.melt(
        df,
        id_vars=["ID"],
        value_vars=["BACKLOG", "NEW", "APPROVED", "IN PROGRESS", 
                    "ANALIZED", "COMMITTED/RESOLVED", "REVIEW", "DONE"],
        var_name="concept:name",
        value_name="time:timestamp"
    )
    df_melted.dropna(subset=["time:timestamp"], inplace=True)
    df_melted["time:timestamp"] = pd.to_datetime(df_melted["time:timestamp"])
    df_melted.rename(columns={"ID": "case_id"}, inplace=True)
    df_melted["case_id"] = df_melted["case_id"].astype(str)
    df_melted["concept:name"] = df_melted["concept:name"].astype(str)
    return df_melted

def convert_to_event_log(df):
    df.rename(columns={
        "case_id": "case:concept:name",
        "concept:name": "concept:name",
        "time:timestamp": "time:timestamp"
    }, inplace=True)
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    return pm4py.convert_to_event_log(df)

def apply_heuristic_miner(log):
    return heuristics_miner.apply_heu(log)

def visualize_heuristic_net(heuristic_net):
    gviz = hn_visualizer.apply(heuristic_net)
    hn_visualizer.view(gviz)  # Opens visualization externally

def apply_inductive_miner(log):
    process_tree = inductive_miner.apply(log)
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    net, initial_marking, final_marking = pt_converter.apply(process_tree, variant=pt_converter.Variants.TO_PETRI_NET)
    return net, initial_marking, final_marking

def visualize_petrinet(net, initial_marking, final_marking):
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)

def perform_conformance_checking(log, net, initial_marking, final_marking):
    replay_result = token_replay.apply(log, net, initial_marking, final_marking)
    if isinstance(replay_result, list) and len(replay_result) > 0:
        avg_fitness = sum(item.get('trace_fitness', 0) for item in replay_result) / len(replay_result)
    else:
        avg_fitness = None
    return avg_fitness

def analyze_performance(log):
    trace_lengths = [len(trace) for trace in log]
    avg_length = sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0
    return avg_length

def analyze_variants(log):
    variants = variants_filter.get_variants(log)
    return {variant: len(cases) for variant, cases in variants.items()}

# ----------------------------------------------------------------------------------
# Advanced Performance Metrics Functions
# ----------------------------------------------------------------------------------
def analyze_advanced_performance(log):
    throughput_times = []
    waiting_times = []
    for trace in log:
        trace_sorted = sorted(trace, key=lambda event: event["time:timestamp"])
        if trace_sorted:
            start = trace_sorted[0]["time:timestamp"]
            end = trace_sorted[-1]["time:timestamp"]
            throughput_times.append((end - start).total_seconds() / 60.0)  # minutes
            for i in range(1, len(trace_sorted)):
                waiting = (trace_sorted[i]["time:timestamp"] - trace_sorted[i-1]["time:timestamp"]).total_seconds() / 60.0
                waiting_times.append(waiting)
    avg_throughput = sum(throughput_times)/len(throughput_times) if throughput_times else 0
    avg_waiting = sum(waiting_times)/len(waiting_times) if waiting_times else 0
    return throughput_times, waiting_times, avg_throughput, avg_waiting

def plot_histogram(data, title, xlabel):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    return fig

def plot_boxplot(data, title, ylabel):
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig

# ----------------------------------------------------------------------------------
# Streamlit Dashboard Main App
# ----------------------------------------------------------------------------------
st.title("Process Mining Dashboard")
st.write("Upload an Excel file and select an analysis.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file is not None:
    sheet_name = st.text_input("Sheet name", value="Plan1")
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    if st.button("Run Process Mining Analysis"):
        with st.spinner("Processing..."):
            df_prepared = load_and_prepare_data(uploaded_file, sheet_name)
            event_log = convert_to_event_log(df_prepared)
            st.success(f"Event log loaded. Number of cases: {len(event_log)}")
            
            analysis_option = st.selectbox("Select analysis option", 
                                             ["Select", 
                                              "Heuristic Miner (Visualize HeuristicsNet)",
                                              "Inductive Miner (Discover and visualize Petri net)",
                                              "Conformance Checking (Token Replay on Petri net)",
                                              "Performance Analysis (Mean Trace Length)",
                                              "Variant Analysis",
                                              "Advanced Performance Metrics (Throughput & Waiting Times)"])
            if analysis_option != "Select":
                if analysis_option.startswith("Heuristic Miner"):
                    heuristic_net = apply_heuristic_miner(event_log)
                    st.write("Heuristic Miner applied. Visualization will open in a separate window.")
                    visualize_heuristic_net(heuristic_net)
                elif analysis_option.startswith("Inductive Miner"):
                    net, initial_marking, final_marking = apply_inductive_miner(event_log)
                    st.write("Inductive Miner applied. Petri net visualization will open in a separate window.")
                    visualize_petrinet(net, initial_marking, final_marking)
                elif analysis_option.startswith("Conformance Checking"):
                    net, initial_marking, final_marking = apply_inductive_miner(event_log)
                    avg_fitness = perform_conformance_checking(event_log, net, initial_marking, final_marking)
                    st.write("Average Fitness from Conformance Checking:", avg_fitness)
                elif analysis_option.startswith("Performance Analysis"):
                    avg_length = analyze_performance(event_log)
                    st.write("Mean Trace Length:", avg_length)
                elif analysis_option.startswith("Variant Analysis"):
                    variants_info = analyze_variants(event_log)
                    st.write("Variants Analysis:")
                    st.write(variants_info)
                elif analysis_option.startswith("Advanced Performance Metrics"):
                    throughput_times, waiting_times, avg_throughput, avg_waiting = analyze_advanced_performance(event_log)
                    st.write("Average Throughput Time (minutes):", avg_throughput)
                    st.write("Average Waiting Time (minutes):", avg_waiting)
                    st.write("Throughput Time Distribution:")
                    hist_fig = plot_histogram(throughput_times, "Throughput Time Distribution", "Minutes")
                    st.pyplot(hist_fig)
                    st.write("Waiting Time Distribution:")
                    hist_fig2 = plot_histogram(waiting_times, "Waiting Time Distribution", "Minutes")
                    st.pyplot(hist_fig2)
                    st.write("Throughput Time Boxplot:")
                    box_fig = plot_boxplot(throughput_times, "Throughput Time Boxplot", "Minutes")
                    st.pyplot(box_fig)
                    st.write("Waiting Time Boxplot:")
                    box_fig2 = plot_boxplot(waiting_times, "Waiting Time Boxplot", "Minutes")
                    st.pyplot(box_fig2)
else:
    st.info("Awaiting file upload.")

# To run this dashboard:
# To run this dashboard, execute the following command in your terminal:
# streamlit run "/Users/rodrigoalmeidadeoliveira/Library/CloudStorage/GoogleDrive-rodrigoalmeidadeoliveira@gmail.com/Outros computadores/Notebook/Python/Projetos/ProcessMining/process_mining_full-dash-v2.py"