# Deployment block to merge multiple impulses

Note: Implemented for EON projects only. Anomaly detection blocks not tested.

## Usage

By default, the quantized version is used when downloading the C++ libraries. To use float32, add the option `--float32` as an argument.

Similarly by default the EON compiled model is used, if you want to use full tflite then add the option `--full-tflite` and be sure to include a recent version of tensorflow lite compiled for your device architecture in the root of your project in a folder named `tensorflow-lite`

If you need a mix of quantized and float32, you can look at the `dzip.download_model` function call in generate.py and change the code accordingly. 

By default, the block will download cached version of builds. You can force new builds using the `--force-build` option.

### Locally

Retrieve API Keys of your projects and run the generate.py command as follows:

```python generate.py --out-directory output --api-keys ei_0b0e...,ei_acde...```

### Docker

Build the container:
```docker build -t multi-impulse .```

Then run:
```docker run --rm -it -v $PWD:/home multi-impulse --api-keys ei_0b0e...,ei_acde...```

### Custom deployment block

Initialize the custom block - select _Deployment block_ and _Library_ when prompted:
```edge-impulse-blocks init```

Push the block:
```edge-impulse-blocks push```

Then go your Organization and Edit the deployment block with:
* CLI arguments: ```--api-keys ei_0b0e...,ei_acde...```
* Provileged mode: **Enabled**

## Compiling the standalone example

The Makefile is for Desktop environment (macOS/Linux). For embedded targets, you'll need to change the cross-compiler or integrate the multi-impulse inference library within your application.

* Unzip the deploy.zip archive (from output/ directory if running on your laptop)
* Open the source/main.cpp file and fill the raw features arrays corresponding to the project IDs
* Run`./build.sh` to compile
* Run `./app` to check the static inferencing results

