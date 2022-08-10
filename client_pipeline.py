import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import time
import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np


def test_infer(req_id, image_file, model_name, print_output=False):
    with open(image_file, 'rb') as fi:
        image_bytes = fi.read()
    image_bytes = np.array([image_bytes], dtype=np.bytes_)
    # Define model's inputs
    inputs = []
    inputs.append(httpclient.InferInput('IMAGE_BYTES', image_bytes.shape, "BYTES"))
    inputs[0].set_data_from_numpy(image_bytes)
    # Define model's outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('BBOXES'))
    outputs.append(httpclient.InferRequestedOutput('CLASSES'))
    outputs.append(httpclient.InferRequestedOutput('MASKS'))
    outputs.append(httpclient.InferRequestedOutput('SCORES'))
    # Send request to Triton server
    triton_client = httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False)
    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)

    if print_output:
        print(req_id, outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', default='sequential', choices=['sequential', 'concurrent'])
    parser.add_argument('--num-reqs', default='1')
    parser.add_argument('--print-output', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_file = args.image
    model_name = args.model
    mode = args.mode
    n_reqs = int(args.num_reqs)

    if mode == 'sequential':
        for i in tqdm(range(n_reqs)):
            test_infer(i, image_file, model_name, args.print_output)
    elif mode == 'concurrent':
        s = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(test_infer,
                                i,
                                image_file,
                                model_name,
                                args.print_output)
                for i in range(n_reqs)
            ]
            wait(futures)
            for f in futures:
                f.results()
        e = time.time()
        print(n_reqs/(e - s))
