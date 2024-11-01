# Deployment block to merge multiple impulses

Note: Implemented for EON projects only. Anomaly detection blocks not tested.

## Usage

Required flags:
- ```--api-keys <key_1>,<key_2>,<key_n>,...```
List of API keys of projects to include in the muti impulse package

- ```--quantization-map <0/1>,<0/1>,<0/1>,...```
List of the switches for quantization for each of the impulses for which API keys were provided. 0 - NOT quantized; 1 - QUANTIZED.

By default the EON compiled model is used, if you want to use full tflite then add the option `--full-tflite` and be sure to include a recent version of tensorflow lite compiled for your device architecture in the root of your project in a folder named `tensorflow-lite`

By default, the block will download cached version of builds. You can force new builds using the `--force-build` option. If a cached version of the required build is not available, an exception will inform about it.

### Locally

Install the requirements
```pip install -r requirements.txt```

Retrieve API Keys of your projects and run the generate.py command as follows:
```python
generate.py --out-directory ./output \
    --api-keys "ei_0b0e...", "ei_acde..." \
    --quantization-map 1,1
```

This will request quantized version for each of the two projects.

### Docker

Build the container:
```docker build -t multi-impulse .```

Then run:
```docker run --rm -it -v $PWD:/home multi-impulse --api-keys "ei_0b0e...", "ei_acde..."````

### Custom deployment block

Initialize the custom block - select _Deployment block_ and _Library_ when prompted:
```edge-impulse-blocks init```

Push the block:
```edge-impulse-blocks push```

Then go your Organization and Edit the deployment block with:
* CLI arguments: ```--api-keys ei_0b0e...,ei_acde... --quantization-map <0/1>,<0/1>```
* Privileged mode: **Enabled**

## Compiling the standalone example

After the block is finished it generates a `deploy.zip` archive that is a standalone buildable example of using all the impulses in multi-impulse package.

This can be useful to sanity-check that your combined impulses behave as expected.

The Makefile is for Desktop environment (macOS/Linux). For embedded targets, you'll need to change the cross-compiler or integrate the multi-impulse inference library within your application.

1. Unzip the deploy.zip archive (from output/ directory if running on your laptop)
2. Open the source/main.cpp file and fill the raw features arrays corresponding to the project IDs. You can get exampel arrays from raw dsp block output in each project.
3. Run`./build.sh` to compile
4. Run `./app` to check the static inferencing results

