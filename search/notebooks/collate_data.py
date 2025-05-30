import pandas as pd

def analyze_model_metrics(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Define the metrics to analyze
    metrics = ['Number of Calls', 'Time Elapsed', 'IoU', 'Bhattacharyya']

    # Group by 'model type' and 'Number of Vectors', and calculate mean and std
    grouped = df.groupby(['Model Name', 'Number of Vectors'])[metrics].agg(['mean', 'std'])

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Reset index to get a clean DataFrame
    result = grouped.reset_index()

    return result

if __name__ == "__main__":
    # Replace 'data.csv' with your CSV file path
    csv_file_path = 'C:\\Users\\Jake\\Learning-Dynamic-Systems\\search\\notebooks\\radial_search_benchmark_results_good.csv'

    result_df = analyze_model_metrics(csv_file_path)

    # Show the result
    print(result_df)

    # Optionally save to a new CSV
    result_df.to_csv("C:\\Users\\Jake\\Learning-Dynamic-Systems\\search\\notebooks\\radial_search_benchmarks_collated.csv", index=False)