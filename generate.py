import os, sys, argparse, json, tempfile, re, shutil
from zipfile import ZipFile
from EIDownload import EIDownload

parser = argparse.ArgumentParser(description='Multi-impulse transformation block')
parser.add_argument('--projects', type=str, help='List of project IDs separated by a comma', required=False)
parser.add_argument('--tmp-directory', type=str, required=False)
parser.add_argument('--out-directory', type=str, default='out/', required=False)

args, unknown = parser.parse_known_args()

# Get projects API Keys
#projectIDs = args.projects.replace(' ', '').split(',')

## DOWNLOADING LIBS

# We bypass download if we already have projects locally in a tmp directory
if not (args.projects and args.tmp_directory):

    if not os.getenv("MULTI_API_KEYS"):
        raise(Exception('MULTI_API_KEYS env variable not set'))
    apiKeys = os.getenv("MULTI_API_KEYS").replace(' ', '').split(',') # comma between keys

    # verify that the input file exists and create the output directory if needed
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # Create temp directory to store zip files and manipulate files
    # tmp_directory argument used for tests
    if args.tmp_directory:
        if not os.path.exists(args.tmp_directory):
            os.makedirs(args.tmp_directory)
        tmpdir = args.tmp_directory
    else:
        tmpdir = tempfile.mkdtemp()

    project_ids = []
    # Download C++ libs and unzip
    for i in range(len(apiKeys)):
        dzip = EIDownload(api_key = apiKeys[i])
        project_ids += [str(dzip.get_project_id())]

        download_path = os.path.join(tmpdir, str(project_ids[i]))

        os.makedirs(download_path)
        zipfile_path = dzip.download_model(download_path, eon = True, quantized = True)

        with ZipFile(zipfile_path, 'r') as zObject:
            zObject.extractall(download_path)
        os.remove(zipfile_path)

else:
    project_ids = args.projects.split(',')
    tmpdir = args.tmp_directory


## GENERIC FUNCTIONS TO EDIT FILES

# Generic function to add suffix to search patterns in a file
def edit_file(file_path, patterns, suffix):
    print("Editing " + file_path)
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()

        # function to add suffix to search patterns
        def add_suffix(term):
            print("Adding " + suffix + " to " + term.group(0))
            return term.group(0) + suffix

        # Search each pattern in file and call add_suffix
        for pattern in patterns:
            file_content = re.sub(pattern, add_suffix, file_content)

        with open(file_path, 'w') as file:
            file.write(file_content)

        print(f"{file_path} edited")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Function to merge model_variables.h
def merge_model_variables(src_file, dest_file):
    start_str = "const char* ei_classifier_inferencing_categories"
    end_str = "const ei_impulse_t ei_default_impulse"
    insert_line_str = "const ei_impulse_t ei_default_impulse"
    include_line_str = '#include "tflite-model/tflite_learn'
    try:
        # Open the first file for reading
        with open(src_file, 'r') as file1:
            file1_contents = file1.readlines()

        start_line = None
        end_line = None
        include_lines = []
        for i, line in enumerate(file1_contents):
            if include_line_str in line:
                include_lines += [line]
            if start_str in line:
                start_line = i
            if end_str in line:
                end_line = i-1
                break

        if start_line is None or end_line is None:
            raise ValueError("Start or end string not found model_variables.h")

        # Open the second file for reading
        with open(dest_file, 'r') as file2:
            file2_contents = file2.readlines()

        # Find the line number for the insertion string in the second file
        insert_line = None
        insert_include_line = None
        for i, line in enumerate(file2_contents):
            if include_line_str in line:
                insert_include_line = i
            if insert_line_str in line:
                insert_line = i
                break

        if insert_line is None:
            raise ValueError("Insertion string not found in model_variables.h")

        # Extract the portion between start and end lines from the first file
        portion_to_copy = file1_contents[start_line:end_line + 1]
        portion_to_copy[0:0] = ["\n"]
        portion_to_copy += ["\n"]

        # Insert the extracted portions into the second file at the specified lines
        file2_contents[insert_include_line:insert_include_line] = include_lines
        file2_contents[insert_line:insert_line] = portion_to_copy

        # Open the second file for writing and overwrite its contents
        with open(dest_file, 'w') as file2:
            file2.writelines(file2_contents)

        print("Portion copied and inserted successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Function to keep intersection of model_ops_define.h
def merge_model_ops(src_file, dest_file):
    try:
        with open(src_file, 'r') as file1:
            lines_file1 = [line.strip() for line in file1.readlines()]

        with open(dest_file, 'r') as file2:
            lines_file2 = [line.strip() for line in file2.readlines()]

        # Find the intersection of lines
        intersection = [line for line in lines_file1 if line in lines_file2]

        # Write the intersection back to src_file
        with open(dest_file, 'w') as file1:
            for line in intersection:
                file1.write(line + '\n')

        print("Merge model_ops done")

    except FileNotFoundError as e:
        print(f"Error: {e}")


## EDITING FILES

# Use first project as target dir
target_dir = os.path.join(tmpdir, project_ids[0])

for p in project_ids:

    # suffix added to different functions and variables
    suffix = "_" + p

    # 1. Edit compiled files in tflite-model/
    pdir = os.path.join(tmpdir, p, 'tflite-model')
    for f in os.listdir(pdir):
        if "compiled" in f:

            # List of patterns to look for and append with suffix
            # Get learn block ID pattern to add projectID as suffix
            patterns = [
                r"tflite_learn_\d+"
            ]
            edit_file(os.path.join(pdir, f), patterns, suffix)

            # Rename filenames
            new_f = f.replace("_compiled", f"{suffix}_compiled")
            os.rename(os.path.join(pdir, f), os.path.join(pdir, new_f))

            # copy to target_dir (1st project)
            if project_ids.index(p) > 0:
                shutil.copy(os.path.join(pdir, new_f), os.path.join(target_dir, 'tflite-model', new_f))


    # 2. Edit model_variables.h
    f = os.path.join(tmpdir, p, "model-parameters/model_variables.h")

    # Patterns may be missing for anomaly detection blocks 
    patterns = [
        r"tflite_learn_\d+",
        r"tflite_graph_\d+",
        "ei_classifier_inferencing_categories",
        r"ei_dsp_config_\d+",
        "ei_dsp_blocks",
        "ei_learning_blocks",
        r"ei_learning_block_config_\d+",
        "ei_calibration"
    ]
    edit_file(f, patterns, suffix)

    # 3. Merge model_variables.h into 1st project
    if project_ids.index(p) > 0:
        merge_model_variables(f, os.path.join(target_dir, "model-parameters/model_variables.h"))


    # 4. Save intersection of trained_model_ops_define.h files
    if project_ids.index(p) > 0:
        f = os.path.join(tmpdir, p, "tflite-model/trained_model_ops_define.h")
        f2 = os.path.join(target_dir, "tflite-model/trained_model_ops_define.h")
        merge_model_ops(f, f2)


# Copy to output folder
shutil.copytree(target_dir, args.out_directory, dirs_exist_ok=True)
shutil.rmtree(tmpdir)