import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import time
import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np


def test_infer(req_id, image_file, model_name, print_output=False):
    img = np.array(Image.open(image_file))
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    # Define model's inputs
    inputs = []
    inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
    inputs[0].set_data_from_numpy(img)
    # Define model's outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
    outputs.append(httpclient.InferRequestedOutput('classes__1'))
    outputs.append(httpclient.InferRequestedOutput('masks__2'))
    outputs.append(httpclient.InferRequestedOutput('scores__3'))
    outputs.append(httpclient.InferRequestedOutput('shape__4'))
    # Send request to Triton server
    triton_client = httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False)
    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)
    print(req_id)
    if print_output:
        print(outputs)


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
