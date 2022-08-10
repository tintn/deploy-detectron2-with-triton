import io
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            image_bytes = \
                pb_utils.get_input_tensor_by_name(request, "image_bytes").as_numpy()[0]
            pil_img = Image.open(io.BytesIO(image_bytes))
            img = np.array(pil_img)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            out_tensor = pb_utils.Tensor("preprocessed_image", img)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses

