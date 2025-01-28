import pickle
import json

def view_pickle_file(file_path, output_path):
    """
    Load a pickle file, filter out correctly classified entries,
    and write the remaining mismatched entries to a text file.
    """
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Filter out correctly classified entries
    filtered_data = []
    for item in data:
        # Find the final operation in the chain (assuming it's the last one)
        final_operation = item['chain'][-1] if item['chain'] else None
        if final_operation and final_operation['operation_name'] == 'simple_query':
            output, _ = final_operation['parameter_and_conf'][0]
            
            # Keep only mismatched entries
            if not ((item['label'] == 1.0 and output == 'YES') or (item['label'] == 0 and output == 'NO')):
                filtered_data.append(item)

    # Convert filtered data to a JSON string with indentation for readability
    pretty_output = json.dumps(filtered_data, indent=2, ensure_ascii=False)

    # Write the filtered, pretty data to the output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(pretty_output)

if __name__ == "__main__":
    file_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/results/tabfact/final_result.pkl"
    output_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/results/tabfact/final_result.txt"
    view_pickle_file(file_path, output_path)