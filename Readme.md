# **Action Recognition**

https://drive.google.com/file/d/1JbU3UppvqwAsq-2Gy14u_7iypq1sbbxd/view?usp=sharing

https://drive.google.com/drive/folders/1HJZLMpzrqD8AaXrHoXWyeJztSFNVmnjq?usp=sharing

Inference scripts [TensorFlow(Inflated 3D Convnet) and Pytorch(Resnet 3D 18)] for Action Recognition, pretrained on Kinetics 400.
## How to use
***Pytorch*** 

    $ python actionRecognitionPytorch.py --model-path PT_MODEL_PATH --folder-path FOLDER_CONTAINING_VIDEOS_PATH

*Example (to use in this repo)*

    $ python actionRecognitionPytorch.py --model-path r3d_18.pt --folder-path inference_videos/

***TensorFlow*** 

    $ python actionRecognitionTensorflow.py --model-path SAVED_MODEL_FOLDER_PATH --folder-path FOLDER_CONTAINING_VIDEOS_PATH

*Example (to use in this repo)*

    $ python actionRecognitionPytorch.py --model-path i3d-kinetics-400_1 --folder-path inference_videos/

## Explaination

***Extract frames and arrange in array***

	def load_video(path):
		all_frames = []
		cap = cv2.VideoCapture(path)
		if (cap.isOpened() is False):
			print('Error while trying to read video. Please check path again')
		# read until end of video
		while(cap.isOpened()):
			# capture each frame of the video
			ret, frame = cap.read()
			if ret is True:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = transform(image=frame)['image']
				all_frames.append(frame)
			else:
				break
		cap.release()
		return all_frames

Here we read capture the video using OpenCV, extract all the frames and arrange in an array of frames.

***Predict the output ***

##### Pytorch

	def predict(frames, model):
		"""Predict top 2 actions taking in array of frames.

		Args:
			frames ([array]): [array of frames]

		"""
		with torch.no_grad():  # we do not want to backprop any gradients
			input_frames = np.array(frames)
			# add an extra dimension
			input_frames = np.expand_dims(input_frames, axis=0)
			# transpose to get [1, 3, num_clips, height, width]
			input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
			# convert the frames to tensor
			input_frames = torch.tensor(input_frames, dtype=torch.float32)
			input_frames = input_frames.to(device)
			# forward pass to get the predictions
			outputs = model(input_frames)
			# softmax layer to get probablities
			percentage = torch.nn.functional.softmax(outputs, dim=1)[0]
			# get the prediction index
			_, preds = torch.sort(outputs.data, descending=True)
			result = [(
				class_names[idx].strip(), f'{percentage[idx] * 100:5.2f}%')
				 for idx in preds[0][:2]]
			return result

Since in ResNet 3D( https://arxiv.org/abs/1711.11248), we use 3D convolutional layers instead of the 2D convolutional layers, we add an extra layer to our array of frames.
Transpose it accroding to requirement of model, and convert to tensor.
Output and prediction probablities are calculating by passing through a softmax layer.

##### TensorFlow

	def predict(frames, model):
		"""Predict top 2 actions taking in array of frames.

		Args:
			frames ([array]): [array of frames]

		"""
		input_frames = np.array(frames) / 255.0
		# Add a batch axis to the to the sample video.
		model_input = tf.constant(input_frames, dtype=tf.float32)[tf.newaxis, ...]

		logits = model(model_input)['default'][0]
		# softmax layer to get probablities
		probabilities = tf.nn.softmax(logits)

		result = [(
				class_names[idx].strip(), f'{probabilities[idx] * 100:5.2f}%')
				 for idx in np.argsort(probabilities)[::-1][:2]]
		return result

Here, we are using Inflated 3D Convnet pretrained model for inference. We convert our array of frames to a constant tensor. Output and prediction probablities are calculating by passing through a softmax layer.


