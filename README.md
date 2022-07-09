![neuroface_header](https://user-images.githubusercontent.com/83948828/178101505-a6654269-c692-41f1-b508-9ff51068fd3f.jpg)

![Typing SVG](https://readme-typing-svg.herokuapp.com?size=100&color=F7F7F7&center=true&vCenter=true&width=1875&height=100&lines=NeuroFace)

---

NeuroFace is a Python framework containing tools for detection, human face recognition, analysis of human facial expressions and gestures on video stream.

<a name='000'></a>
<h2>Table of Contents</h2>

<ul>
    <ol type='1'>
    <li><a href='#001'>Environment</a></li>
    <li><a href='#002'>Installation</a></li>
    </ol>
</ul>

<a name='001'></a>
<h2>Environment</h2>

- [`torch`](https://github.com/pytorch/pytorch) >= `1.12.0`

  > PyTorch models implementation and deployment.

- [`torchvision`](https://github.com/pytorch/vision) >= `0.13.0`

  > Reading and preprocessing frames as torch tensors.

- [`mediapipe`](https://github.com/google/mediapipe) >= `0.8.10.1`

  > MediaPipe models implementation and deployment.

- [`opencv-python`](https://github.com/opencv/opencv-python) >= `4.6.0.66`

  > Reading and preprocessing frames as numpy ndarrays.

- [`Pillow`](https://github.com/python-pillow/Pillow) >= `9.2.0`

  > Reading and preprocessing frames as PIL Images.

- [`av`](https://github.com/PyAV-Org/PyAV) >= `9.2.0`

  > Binding torchvision to ffmpeg to read streams.

- [`gdown`](https://github.com/wkentaro/gdown) >= `4.5.1`

  > Downloading large files from Google Drive.

<h4>System requirements:</h4>

- [`CUDA`](https://developer.nvidia.com/cuda-downloads) >= `11.3`

  > GPU compute access.

<h4>Quick installation:</h4>

```python
pip install -r requirements.txt
```

<a name='002'></a>
<h2>Installation</h2>

```python
git clone https://github.com/mdvdv/neuroface.git
```
