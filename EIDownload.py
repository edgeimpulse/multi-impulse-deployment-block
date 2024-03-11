# Module to build and download models using EI API

import requests, json, time, re, os

class EIDownload:

    def __init__(self, api_key, project_id = None):
        self.api_key = api_key
        if project_id is None:
            self.project_id = self.set_project_id()
            print("Project ID is " + str(self.project_id))
        else:
            self.project_id = project_id

    def get_project_id(self):
        return self.project_id
    
    def set_project_id(self):
        url = f"https://studio.edgeimpulse.com/v1/api/projects"
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.request("GET", url, headers=headers)
        body = json.loads(response.text)
        if (not body['success']):
            raise Exception(body['error'])
        
        return body['projects'][0]['id']


    def download_model(self, out_directory, eon=True, quantized=True, force_build=False):
        if self.project_id is None:
            raise Exception('Project ID is not set')

        if eon:
            engine = 'tflite-eon'
        else:
            engine = 'tflite'
        if quantized:
            model_type = 'int8'
        else:
            model_type = 'float32'
    
        # Check if build is available first
        if force_build or not self.build_available(engine, model_type):     
            print("No build artefact found for project " + str(self.project_id) + ", will build library first.")  
            job_id = self.build_model(engine, model_type)
            self.wait_for_job_completion(job_id)
            print('Build OK')

        url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}/deployment/download"
        querystring = {
            "type": "zip",
            "modelType": model_type,
            "engine": engine
        }
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/zip",
            "Content-Type": "application/json",
        }
        response = requests.request("GET", url, headers=headers, params=querystring)

        d = response.headers['Content-Disposition']
        fname = re.findall("filename\*?=(.+)", d)[0].replace('utf-8\'\'', '')

        with open(os.path.join(out_directory, fname), 'wb') as f:
            f.write(response.content)
        print('Export ZIP saved in: ' + os.path.join(out_directory, fname) + ' (' + str(len(response.content)) + ' Bytes)')
        
        return os.path.join(out_directory, fname)
    
    def build_available(self, engine, model_type):
        url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}/deployment"
        querystring = {"type": "zip", "modelType": model_type, "engine": engine}
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.request("GET", url, headers=headers, params=querystring)
        body = json.loads(response.text)
        if (not body['success']):
            raise Exception(body['error'])
        
        return body['hasDeployment']


    def build_model(self, engine, model_type):
        url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}/jobs/build-ondevice-model"
        querystring = {"type": "zip"}
        payload = {"engine": engine, "modelType": model_type}
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        response = requests.request("POST", url, json=payload, headers=headers, params=querystring)
        body = json.loads(response.text)
        if (not body['success']):
            raise Exception(body['error'])
        return body['id']

    def get_stdout(self, job_id, skip_line_no):
        url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}/jobs/{job_id}/stdout"
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        response = requests.request("GET", url, headers=headers)
        body = json.loads(response.text)
        if (not body['success']):
            raise Exception(body['error'])
        stdout = body['stdout'][::-1] # reverse array so it's old -> new
        return [ x['data'] for x in stdout[skip_line_no:] ]

    def wait_for_job_completion(self, job_id):
        skip_line_no = 0

        url = f"https://studio.edgeimpulse.com/v1/api/{self.project_id}/jobs/{job_id}/status"
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        while True:
            response = requests.request("GET", url, headers=headers)
            body = json.loads(response.text)
            if (not body['success']):
                raise Exception(body['error'])

            stdout = self.get_stdout(job_id, skip_line_no)
            for l in stdout:
                print(l, end='')
            skip_line_no = skip_line_no + len(stdout)

            if (not 'finished' in body['job']):
                print('Still building...')
                time.sleep(1)
                continue
            if (not body['job']['finishedSuccessful']):
                raise Exception('Job failed')
            else:
                break


