import os
import json
import re


def create_jsonl_from_folder(input_folder, output_file):
    """
    Reads all JSON files from an input folder, corrects their formatting,
    renames keys, and writes the data as a JSON Lines (.jsonl) file.

    Each entry from the source files will become a separate line in the
    output file.
    """
    print(f"\nReading files from '{input_folder}' to create JSONL output...")

    if not os.path.isdir(input_folder):
        print(f"\nError: Input folder '{input_folder}' not found.")
        return

    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])

    if not json_files:
        print(f"\nWarning: No JSON files found in '{input_folder}'.")
        return

    total_entries = 0
    # Open the output file once to write all entries line by line
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for filename in json_files:
                file_path = os.path.join(input_folder, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        content = f_in.read()

                    # Fix malformed JSON by adding commas and wrapping in a list
                    content = re.sub(r'}\s*{', '}, {', content)
                    valid_json_string = f'[{content}]'
                    
                    list_of_data = json.loads(valid_json_string)

                    # Process each dictionary (entry) from the file
                    for data in list_of_data:

                        new_entry = {
                            "question": data.get("user_input", ""),
                            "contexts": data.get("reference_contexts", []),
                            "answer": data.get("reference", "")
                        }

                        # Write the newly formatted entry as one line in the output file
                        f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                        total_entries += 1

                except json.JSONDecodeError as e:
                    print(f"\nError decoding JSON from file '{filename}'. Please check for syntax errors. Details: {e}")
                except Exception as e:
                    print(f"\nAn unexpected error occurred with file '{filename}': {e}")
        
        print(f"\nSuccessfully created '{output_file}' with {total_entries} total entries.")

    except Exception as e:
        print(f"\nAn error occurred while writing to the output file: {e}")


def setup_and_run():
    """Sets up the necessary directories and runs the main processing function."""
    input_dir = "./data/synthetic_test_set/generated_sts_files"
    output_dir = "./data/synthetic_test_set/"
    output_filepath = os.path.join(output_dir, "combined_sts.jsonl")

    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the main function
    create_jsonl_from_folder(input_dir, output_filepath)


if __name__ == "__main__":
    setup_and_run()