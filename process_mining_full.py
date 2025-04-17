import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils

# Heuristic Miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# Inductive Miner and Petri net visualization
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# Conformance checking (token replay)
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# Performance analysis (trace statistics)
from pm4py.statistics.traces.generic.log import case_statistics

# Variant analysis
from pm4py.algo.filtering.log.variants import variants_filter


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
    hn_visualizer.view(gviz)


def apply_inductive_miner(log):
    return inductive_miner.apply(log)


def visualize_petrinet(net, initial_marking, final_marking):
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)


def perform_conformance_checking(log, net, initial_marking, final_marking):
    replay_result = token_replay.apply(log, net, initial_marking, final_marking)
    print("Conformance Checking Results:")
    print("Fitness:", replay_result.get('fitness', 'N/A'))
    return replay_result


def analyze_performance(log):
    mean_trace_length = case_statistics.get_mean_trace_length(log)
    print("Mean Trace Length:", mean_trace_length)
    return mean_trace_length


def analyze_variants(log):
    variants = variants_filter.get_variants(log)
    print("Variants Analysis:")
    for variant, cases in variants.items():
        print(f"{variant}: {len(cases)} cases")
    return variants


def main():
    file_path = 'métricas/métricas_bs2_30092019.xlsx'
    sheet_name = "Plan1"
    
    df_prepared = load_and_prepare_data(file_path, sheet_name)
    event_log = convert_to_event_log(df_prepared)
    print("Event log loaded. Number of cases:", len(event_log))
    
    print("\nSelect an analysis option:")
    print("1. Heuristic Miner (Visualize HeuristicsNet)")
    print("2. Inductive Miner (Discover and visualize Petri net)")
    print("3. Conformance Checking (Token Replay on Petri net)")
    print("4. Performance Analysis (Mean Trace Length)")
    print("5. Variants Analysis")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        heuristic_net = apply_heuristic_miner(event_log)
        visualize_heuristic_net(heuristic_net)
    elif choice == '2':
        net, initial_marking, final_marking = apply_inductive_miner(event_log)
        visualize_petrinet(net, initial_marking, final_marking)
    elif choice == '3':
        net, initial_marking, final_marking = apply_inductive_miner(event_log)
        perform_conformance_checking(event_log, net, initial_marking, final_marking)
    elif choice == '4':
        analyze_performance(event_log)
    elif choice == '5':
        analyze_variants(event_log)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
