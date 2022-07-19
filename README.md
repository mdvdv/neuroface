![neuroface_header](https://user-images.githubusercontent.com/83948828/178101505-a6654269-c692-41f1-b508-9ff51068fd3f.jpg)

![Typing SVG](https://readme-typing-svg.herokuapp.com?size=100&color=BBBBBB&center=true&vCenter=true&width=1875&height=100&lines=NeuroFace)

---

NeuroFace is a Python framework containing tools for detection, human face recognition, analysis of human facial expressions and gestures on video.

<a name='000'></a>
<h2>Table of Contents</h2>

<ul>
    <ol type='1'>
        <li><a href='#001'>Environment</a></li>
        <li><a href='#002'>Installation</a></li>
        <li><a href='#003'>Supported Models</a></li>
        <li><a href='#004'>References</a></li>
    </ol>
</ul>

<a name='001'></a>
<h2>1. Environment</h2>

- [`torch`](https://github.com/pytorch/pytorch) >= `1.12.0`: PyTorch models implementation and deployment.

- [`torchvision`](https://github.com/pytorch/vision) >= `0.13.0`: Reading and preprocessing frames as torch tensors.

- [`mediapipe`](https://github.com/google/mediapipe) >= `0.8.10.1`: MediaPipe models implementation and deployment.

- [`opencv-python`](https://github.com/opencv/opencv-python) >= `4.6.0.66`: Reading and preprocessing frames as numpy ndarrays.

- [`Pillow`](https://github.com/python-pillow/Pillow) >= `9.2.0`: Reading and preprocessing frames as PIL Images.

- [`av`](https://github.com/PyAV-Org/PyAV) >= `9.2.0`: Binding torchvision to ffmpeg to read streams.

- [`gdown`](https://github.com/wkentaro/gdown) >= `4.5.1`: Downloading large files from Google Drive.

<h4>System requirements:</h4>

- [`CUDA`](https://developer.nvidia.com/cuda-downloads) >= `11.3`: GPU compute access.

<h4>Quick installation:</h4>

```python
pip install -r requirements.txt
```

<a name='002'></a>
<h2>2. Installation</h2>

```python
git clone https://github.com/mdvdv/neuroface.git
```

<a name='003'></a>
<h2>3. Supported Models</h2>

<h3>Face Embedding</h3>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Training Dataset</th>
      <th>Backbone</th>
      <th>LWF Metric</th>
      <th>Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Inception-ResNet V1</b></td>
      <td>CASIA-WebFace</td>
      <td>Inception</td>
      <td>0.9905</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1rgLytxUaOUrtjpxCl-mQFGYdUfSWgQCo'>20180408-102900-casia-webface.pt</a></td>
    </tr>
    <tr>
      <td><b>Inception-ResNet V1</b></td>
      <td>VGGFace2</td>
      <td>Inception</td>
      <td>0.9965</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1P4OqfwcUXXuycmow_Fb8EXqQk5E7-H5E'>20180402-114759-vggface2.pt</a></td>
    </tr>
  </tbody>
</table>

<h3>Facial Landmark Detection</h3>

<a name='004'></a>
<h2>4. References</h2>

<ul>
    <ol type='1'>
        <li>Face Recognition Using Pytorch: https://github.com/timesler/facenet-pytorch.</a></li>
        <li>Cross-platform, Customizable ML Solutions for Live and Streaming Media: https://github.com/google/mediapipe.</a></li>
        <li>Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition: https://github.com/yaoing/dan.</a></li>
    </ol>
</ul>
