import json
import os
import re


def create_questions_txt_from_folder(docs_folder_path, txt_file_path):
    """
    Reads all .json files from a folder. Each file can contain multiple 
    concatenated JSON objects. It extracts the 'user_input' string from 
    each object and writes them to a single text file.
    """
    all_questions = []

    if not os.path.isdir(docs_folder_path):
        print(f"\nError: Directory '{docs_folder_path}' not found.")
        return

    for filename in os.listdir(docs_folder_path):
        if filename.endswith(('.json', '.jsonl')):
            json_file_path = os.path.join(docs_folder_path, filename)
            
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    # Read the entire file content
                    content = f.read()

                # Replace the gap between objects `} {` with `},{`
                # The regex `r'}\s*{'` finds a `}` followed by any whitespace `\s*` followed by a `{`
                modified_content = re.sub(r'}\s*{', '},{', content.strip())

                # Wrap the whole string in brackets to make it a valid JSON array
                json_array_string = f"[{modified_content}]"
                
                # Parse the newly formed JSON array string
                data_list = json.loads(json_array_string)
                
                questions_in_file = []
                # Iterate through the list of objects we just parsed
                for data_object in data_list:
                    question = data_object.get('user_input')
                    if isinstance(question, str):
                        questions_in_file.append(question)
                
                if questions_in_file:
                    all_questions.extend(questions_in_file)
                    print(f"\nFound {len(questions_in_file)} questions in '{filename}'")
                else:
                    print(f"\nWarning: No 'user_input' strings found in '{filename}'.")

            except json.JSONDecodeError as e:
                print(f"\nWarning: Could not decode JSON from '{filename}'. It may have a formatting error. Details: {e}. Skipping.")
            except Exception as e:
                print(f"\nAn unexpected error occurred with file '{filename}': {e}")

    # After checking all files, write the aggregated questions to the output file
    if all_questions:
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_questions))
            print(f"\nSuccessfully extracted a total of {len(all_questions)} questions to '{txt_file_path}'")
        except Exception as e:
            print(f"\nError writing to output file '{txt_file_path}': {e}")
    else:
        print("\nNo questions were found in any of the JSON files.")


if __name__ == "__main__":
    input_folder = './data/synthetic_test_set/generated_sts_files'
    output_questions_file = './data/processed/question_answer_pairs/questions.txt'
    
    create_questions_txt_from_folder(input_folder, output_questions_file)