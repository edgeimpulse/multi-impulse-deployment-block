import os, argparse, tempfile, re, shutil
from zipfile import ZipFile
from EIDownload import EIDownload
from utils import *
import logging

parser = argparse.ArgumentParser(description='Multi-impulse transformation block')
parser.add_argument('--api-keys', type=str, help='List of API Keys', required=False)
parser.add_argument('--projects', type=str, help='List of project IDs separated by a comma', required=False)
parser.add_argument('--tmp-directory', type=str, required=False)
parser.add_argument('--out-directory', type=str, default='/home/output', required=False)
parser.add_argument("--force-build", action="store_true", help="Force build libraries, no cache")
parser.add_argument("--engine", type=str, choices = ['eon', 'tflite'], default='eon', help="Inferencing engine to use.")
parser.add_argument("--quantization-map", type=str, help="Description of quantization policy for each impulse", required=False)

# EG
# --api-keys apiA,apiB \
# --quantization-map 0,1
# This means that the first impulse of the first project will NOT be quantized and the second impulse WILL be quantized

args, unknown = parser.parse_known_args()

logging.basicConfig()
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Get projects API Keys
#projectIDs = args.projects.replace(' ', '').split(',')

## DOWNLOADING LIBS

# We bypass download if we already have projects locally in a tmp directory
if not (args.projects and args.tmp_directory):

    if not args.api_keys:
        raise(Exception('--api-keys argument not set'))
    apiKeys = args.api_keys.replace(' ', '').split(',') # comma between keys

    if not args.quantization_map:
        raise(Exception('--quantization-map argument not set'))
    quantizationMap = args.quantization_map.replace(' ', '').split(',') # comma between mask

    # check for duplicate projects
    apiKeysSet = list(set(apiKeys))
    if len(apiKeysSet) != len(apiKeys):
        raise(Exception('Duplicate projects detected. Please provide unique API keys'))

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
        if quantizationMap[i] == '0':
            quantized = False
        else:
            quantized = True

        zipfile_path = dzip.download_model(download_path, eon = (args.engine == 'eon'), quantized = quantized, force_build = args.force_build)

        with ZipFile(zipfile_path, 'r') as zObject:
            zObject.extractall(download_path)
        os.remove(zipfile_path)

else:
    project_ids = args.projects.split(',')
    tmpdir = args.tmp_directory

## EDITING FILES

# create a target dir
target_dir = os.path.join(args.out_directory, "output")
# copy from the first project
shutil.copytree(os.path.join(tmpdir, project_ids[0]), target_dir, dirs_exist_ok=True)
include_lines = []

# Save intersection of trained_model_ops_define.h files
f1 = os.path.join(tmpdir, project_ids[0], "tflite-model/trained_model_ops_define.h")
f2 = os.path.join(tmpdir, project_ids[1], "tflite-model/trained_model_ops_define.h")
merge_model_ops(f1, f2)
shutil.copy(f2, os.path.join(target_dir, "tflite-model/trained_model_ops_define.h"))

# merge the resolvers if tflite
if args.engine == 'tflite':
    f1 = os.path.join(tmpdir, project_ids[0], "tflite-model/tflite-resolver.h")
    f2 = os.path.join(tmpdir, project_ids[1], "tflite-model/tflite-resolver.h")
    merge_tflite_resolver(f1, f2)
    shutil.copy(f2, os.path.join(target_dir, "tflite-model/tflite-resolver.h"))

# merge the model metadata
f1 = os.path.join(tmpdir, project_ids[0], "model-parameters/model_metadata.h")
f2 = os.path.join(tmpdir, project_ids[1], "model-parameters/model_metadata.h")
merge_model_metadata(f1, f2)
shutil.copy(f2, os.path.join(target_dir, "model-parameters/model_metadata.h"))

for p in project_ids:

    # suffix added to different functions and variables
    suffix = "_" + p
    logger.info(f"Processing Project{str(suffix)}")

    # Edit compiled files in tflite-model/
    model_dir = os.path.join(tmpdir, p, 'tflite-model')
    for f in os.listdir(model_dir):
        if "compiled" in f:

            # List of patterns to look for and append with suffix
            # Get learn block ID pattern to add projectID as suffix
            patterns = [
                r"tflite_learn_\d+"
            ]
            edit_file(os.path.join(model_dir, f), patterns, suffix)

            # Rename filenames
            new_f = f.replace("_compiled", f"{suffix}_compiled")
            os.rename(os.path.join(model_dir, f), os.path.join(model_dir, new_f))

            # copy to target_dir (1st project)
            if project_ids.index(p) > 0:
                shutil.copy(os.path.join(model_dir, new_f), os.path.join(target_dir, 'tflite-model', new_f))

        elif f.startswith("tflite_learn_"):
            patterns = [
                r"tflite_learn_\d+"
            ]
            edit_file(os.path.join(model_dir, f), patterns, suffix)

            # Rename filenames
            name, ext = os.path.splitext(f)
            new_f = f"{name}{suffix}{ext}"
            os.rename(os.path.join(model_dir, f), os.path.join(model_dir, new_f))

            # copy to target_dir (1st project)
            if project_ids.index(p) > 0:
                shutil.copy(os.path.join(model_dir, new_f), os.path.join(target_dir, 'tflite-model', new_f))

    # Edit model_variables.h
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
        r"ei_learning_block_\d+_inputs",
        "ei_object_detection_nms(?!_config)",
        "ei_calibration"
    ]
    edit_file(f, patterns, suffix)

    # Merge model_variables.h into 1st project
    if project_ids.index(p) > 0:
        merge_model_variables(f, os.path.join(target_dir, "model-parameters/model_variables.h"))

# Copy template files to tmpdir
shutil.copytree('templates', target_dir, dirs_exist_ok=True)

# Get sample code to customize main.cpp

# Get impulses ID from model_variables.h
with open(os.path.join(target_dir, 'model-parameters/model_variables.h'), 'r') as file:
    file_content = file.read()
impulses_id_set = set(re.findall(r"impulse_(\d+)_(\d+)", file_content))
impulses_id = {}
for i in impulses_id_set:
    impulses_id[i[0]] = i[1]

get_signal_code = "\n"
raw_features_code = "\n"
run_classifier_code = "\n"
callback_function_code = "\n"
newline = "\n"

# custom code for each project
for p in project_ids:
    get_signal_code += f"static int get_signal_data_{p}(size_t offset, size_t length, float *out_ptr);{newline}"
    raw_features_code += f"static const float features_{p}[] = {{ ... }}; // copy features from project {p}{newline}"

    deploy_version = impulses_id[p]
    run_classifier_code += f"""
    // new process_impulse call for project ID {p}
    signal.total_length = impulse_{p}_{deploy_version}.dsp_input_frame_size;
    signal.get_data = &get_signal_data_{p};
    res = process_impulse(&impulse_handle_{p}_{deploy_version}, &signal, &result, false);
    printf("process_impulse for project {p} returned: %d\\r\\n", res);
    display_custom_results(&result, &impulse_{p}_{deploy_version});
    {newline}"""

    callback_function_code += f"""
static int get_signal_data_{p}(size_t offset, size_t length, float *out_ptr) {{
    for (size_t i = 0; i < length; i++) {{
        out_ptr[i] = (features_{p} + offset)[i];
    }}
    return EIDSP_OK;
}}
{newline}"""

# Insert custom code in main.cpp
with open(os.path.join(target_dir, 'source/main.cpp'), 'r') as file1:
    main_template = file1.readlines()

idx = main_template.index("// get_signal declaration inserted here\n") +1
main_template[idx:idx] = get_signal_code
idx = main_template.index("// raw features array inserted here\n") + 1
main_template[idx:idx] = raw_features_code
idx = main_template.index("// process_impulse inserted here\n") + 1
main_template[idx:idx] = run_classifier_code
idx = main_template.index("// callback functions inserted here\n") + 1
main_template[idx:idx] = callback_function_code

logger.info("Editing main.cpp")
with open(os.path.join(target_dir, 'source/main.cpp'), 'w') as file1:
    file1.writelines(main_template)
logger.info("main.cpp edited")

logger.info("Merging done!")

# Create archive
shutil.make_archive(os.path.join(args.out_directory, 'deploy'), 'zip', target_dir)
