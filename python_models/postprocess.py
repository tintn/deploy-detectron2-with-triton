from PIL import Image
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def get_inputs(self, request):
        input_tensor_names = \
            ['bboxes', 'classes', 'masks', 'scores', 'shape']
        inputs = {
            tensor_name: pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()
                  for tensor_name in input_tensor_names
        }
        return inputs

    def paste_mask(self, mask, box, img_h, img_w, threshold):
        """
        Paste raw masks with fixed resolution from the mask head to an image
        NOTE: You can find the better implementation from:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/mask_ops.py

        This method largely based on "paste_mask_in_image_old" from mask_ops.py.
        I used it for the sake of simplicity.

        Args:
            mask: M x M array where M is the Pooler resolution of your mask head
            box: array of shape (4,)
            img_h, img_w (int): Image height and width.
            threshold (float): Mask binarization threshold in [0, 1].
        Return:
            im_mask (Tensor):
                The resized and binarized object mask pasted into the original
                image plane (a tensor of shape (img_h, img_w)).
        """
        box = box.astype(np.int)
        # Resize the mask to the size of the bbox
        samples_w = box[2] - box[0] + 1
        samples_h = box[3] - box[1] + 1
        mask = Image.fromarray(mask)
        mask = mask.resize((samples_w, samples_h), resample=Image.BILINEAR)
        mask = np.array(mask, copy=False)
        mask = np.array(mask > threshold, dtype=np.uint8)

        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, img_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, img_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])
        ]
        return im_mask

    def postprocess(self, predictions):
        img_h, img_w = predictions.pop('shape')
        # Filter out predictions with low confidence scores
        scores = predictions['scores']
        predictions = {name: tensor[scores > 0.5, ...] for name, tensor in predictions.items()}
        # Paste masks to the full image
        full_masks = [
            self.paste_mask(mask[0, :, :], box, img_h, img_w, 0.5)
            for mask, box in zip(predictions['masks'], predictions['bboxes'])
        ]
        predictions['masks'] = np.stack(full_masks, axis=0)
        return predictions

    def execute(self, requests):
        responses = []
        for request in requests:
            predictions = self.get_inputs(request)
            predictions = self.postprocess(predictions)
            # prepare outputs for the reponse
            out_tensors = []
            for name in ['bboxes', 'classes', 'scores', 'masks']:
                tensor = pb_utils.Tensor('post_' + name, predictions[name])
                out_tensors.append(tensor)
            response = pb_utils.InferenceResponse(output_tensors=out_tensors)
            responses.append(response)
        return responses
