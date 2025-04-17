import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.heuristics.variants import classic


def load_and_prepare_data(file_path, sheet_name):
    """
    Load and prepare the data for process mining.
    """
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Transform activity columns into a suitable format for process mining
    df_melted = pd.melt(
        df,
        id_vars=["ID"],
        value_vars=[
            "BACKLOG", "NEW", "APPROVED", "IN PROGRESS", "ANALIZED",
            "COMMITTED/RESOLVED", "REVIEW", "DONE"
        ],
        var_name="concept:name",
        value_name="time:timestamp"
    )

    # Remove rows without timestamps and convert to datetime
    df_melted.dropna(subset=["time:timestamp"], inplace=True)
    df_melted["time:timestamp"] = pd.to_datetime(df_melted["time:timestamp"])

    # Rename columns for PM4Py compatibility
    df_melted.rename(columns={"ID": "case_id"}, inplace=True)

    # Convert columns to string for PM4Py compatibility
    df_melted["case_id"] = df_melted["case_id"].astype(str)
    df_melted["concept:name"] = df_melted["concept:name"].astype(str)

    return df_melted


def convert_to_event_log(df):
    """
    Convert the prepared DataFrame into an event log for PM4Py.
    """
    # Ensure the DataFrame has the correct column names for PM4Py
    df.rename(columns={
        "case_id": "case:concept:name",
        "concept:name": "concept:name",
        "time:timestamp": "time:timestamp"
    }, inplace=True)

    # Convert timestamp columns to datetime (if not already done)
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    # Convert to event log
    return pm4py.convert_to_event_log(df)  

def apply_heuristic_miner(log):
    """
    Apply the Heuristic Miner algorithm to the event log.
    """
    # Apply the Heuristic Miner algorithm to get a HeuristicsNet
    heuristic_net = heuristics_miner.apply_heu(log)
    return heuristic_net

def visualize_heuristic_net(heuristic_net):
    """
    Visualize the heuristic net using PM4Py.
    """
    gviz = hn_visualizer.apply(heuristic_net)
    hn_visualizer.view(gviz)


if __name__ == "__main__":
    # File path and sheet name
    file_path = 'métricas/métricas_bs2_30092019.xlsx'
    sheet_name = "Plan1"

    # Load and prepare data
    df_prepared = load_and_prepare_data(file_path, sheet_name)

    # Convert to event log
    event_log = convert_to_event_log(df_prepared)

    # Apply Heuristic Miner
    heuristic_net = apply_heuristic_miner(event_log)

    # Visualize the heuristic net
    visualize_heuristic_net(heuristic_net)

