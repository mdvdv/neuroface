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
        <ol>
            <li><a href='#031'>Face Detection</a></li>
            <li><a href='#032'>Face Embedding</a></li>
            <li><a href='#033'>Facial Landmark Detection</a></li>
            <li><a href='#034'>Pose Landmark Detection</a></li>
            <li><a href='#035'>Facial Expression Recognition</a></li>
        </ol>
        <li><a href='#004'>Usage</a></li>
        <li><a href='#005'>References</a></li>
    </ol>
</ul>

<a name='001'></a>
<h2>1. Environment</h2>

- [`torch`](https://github.com/pytorch/pytorch) >= `1.12.0`: PyTorch models implementation and deployment.

- [`torchvision`](https://github.com/pytorch/vision) >= `0.13.0`: Reading and preprocessing frames as PyTorch tensors.

- [`mediapipe`](https://github.com/google/mediapipe) >= `0.8.10.1`: MediaPipe models implementation and deployment.

- [`opencv-python`](https://github.com/opencv/opencv-python) >= `4.6.0.66`: Reading and preprocessing frames as NumPy arrays.

- [`Pillow`](https://github.com/python-pillow/Pillow) >= `9.2.0`: Reading and preprocessing frames as PIL Images.

- [`av`](https://github.com/PyAV-Org/PyAV) >= `9.2.0`: Binding Torchvision to FFmpeg to read streams.

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

<details open>
<summary><h3><a name='031'>3.1 Face Detection</a></h3></summary>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Training Dataset</th>
      <th>Backbone</th>
      <th><abbr title='Average Precision'>AP</abbr> Metric</th>
      <th>Weights</th>
      <td>Paper</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan='3'><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/face/detection/mtcnn.py'>MTCNN</a></b></td>
      <td rowspan='4'><a href='https://arxiv.org/pdf/1511.06523v1.pdf'>WiderFace</a></td>
      <td>P-Net</td>
      <td>0.946</td>
      <td><a href='https://drive.google.com/uc?export=view&id=11il5MJc7VRdpiU_HdstX9Gczdxdb_0M8'>P-Net</a> (28 KB)</td>
      <td rowspan='3'><a href='https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf'>PDF</a></td>
    </tr>
    <tr>
      <td>R-Net</td>
      <td>0.954</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1ykKHaW6or-bWSgCGXJYV3F2B9vU6U3aM'>R-Net</a> (394 KB)</td>
    </tr>
    <tr>
      <td>O-Net</td>
      <td>0.954</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1NDE8q3O741FW960GDxBnuSkPJS3mugfh'>O-Net</a> (1.5 MB)</td>
    </tr>
    <tr>
      <td><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/face/detection/retinaface.py'>RetinaFace</a></b></td>
      <td>MobileNet V1</td>
      <td>0.914</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1-AxXlAFoE5KHBy3ugoi3oi9r-X1hYK_B'>RetinaFace</a> (1.7 MB)</td>
      <td><a href='https://arxiv.org/pdf/1905.00641'>PDF</a></td>
    </tr>
    <tr>
      <td><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/face/detection/mediapipe_face.py'>MediaPipe Face</a></b></td>
      <td>-</td>
      <td>BlazeFace</td>
      <td>0.9861</td>
      <td>-</td>
      <td><a href='https://arxiv.org/pdf/1907.05047'>PDF</a></td>
    </tr>
  </tbody>
</table>
</details>

<details open>
<summary><h3><a name='032'>3.2 Face Embedding</a></h3></summary>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Training Dataset</th>
      <th>Backbone</th>
      <th><abbr title='Average Precision'>AP</abbr> Metric</th>
      <th>Weights</th>
      <td>Paper</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan='2'><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/face/comparison/inception_resnet_v1.py'>Inception-ResNet V1</a></b></td>
      <td><a href='https://arxiv.org/pdf/1411.7923'>CASIA-WebFace</a></td>
      <td>Inception</td>
      <td>0.9905</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1rgLytxUaOUrtjpxCl-mQFGYdUfSWgQCo'>CASIA-WebFace</a> (110.5 MB)</td>
      <td rowspan='2'><a href='https://arxiv.org/pdf/1503.03832.pdf'>PDF</a></td>
    </tr>
    <tr>
      <td><a href='http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf'>VGGFace2</a></td>
      <td>Inception</td>
      <td>0.9965</td>
      <td><a href='https://drive.google.com/uc?export=view&id=1P4OqfwcUXXuycmow_Fb8EXqQk5E7-H5E'>VGGFace2</a> (106.7 MB)</td>
    </tr>
  </tbody>
</table>
</details>

<details open>
<summary><h3><a name='033'>3.3 Facial Landmark Detection</a></h3></summary>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Backbone</th>
      <th><abbr title='Mean Absolute Distance'>MAD</abbr> Metric</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/landmarks/face_mesh.py'>MediaPipe Face Mesh</a></b></td>
      <td>BlazeFace</td>
      <td>0.396</td>
      <td><a href='https://arxiv.org/pdf/1907.06724'>PDF</a></td>
    </tr>
  </tbody>
</table>
</details>

<details open>
<summary><h3><a name='034'>3.4 Pose Landmark Detection</a></h3></summary>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Backbone</th>
      <th colspan="2"><abbr title='Mean Squared Distance'>MSE</abbr> Metric</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/landmarks/hands.py'>MediaPipe Hands</a></b></td>
      <td rowspan="2">BlazePalm</td>
      <td>Light</td>
      <td>11.83</td>
      <td rowspan="2"><a href='https://arxiv.org/pdf/2006.10214'>PDF</a></td>
    </tr>
    <tr>
      <td>Full</td>
      <td>10.05</td>
    </tr>
  </tbody>
</table>
</details>

<details open>
<summary><h3><a name='035'>3.5 Facial Expression Recognition</a></h3></summary>

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Training Dataset</th>
      <th>Backbone</th>
      <th><abbr title='Average Precision'>AP</abbr> Metric</th>
      <th>Weights</th>
      <td>Paper</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href='https://github.com/mdvdv/neuroface/blob/main/neuroface/emotions/attention_resnet.py'>DAN</a></b></td>
      <td><a href='https://arxiv.org/pdf/1708.03985v4.pdf'>AffectNet-8</a></td>
      <td>ResNet</td>
      <td>0.6209</td>
      <td><a href='https://drive.google.com/uc?export=view&id=17lzsrHyuSGd2cZuNHdAAPCw6JsrjgFIn'>AffectNet-8</a> (226 MB)</td>
      <td><a href='https://arxiv.org/pdf/2109.07270.pdf'>PDF</a></td>
    </tr>
  </tbody>
</table>
</details>

<a name='004'></a>
<h2>4. Usage</h2>

Code execution examples are presented in the [documentation](https://github.com/mdvdv/neuroface/blob/main/docs/README.md) section.

<a name='005'></a>
<h2>5. References</h2>

<ul>
    <ol type='1'>
        <li>Face Recognition Using Pytorch: https://github.com/timesler/facenet-pytorch.</a></li>
        <li>Cross-platform, Customizable ML Solutions for Live and Streaming Media: https://github.com/google/mediapipe.</a></li>
        <li>Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition: https://github.com/yaoing/dan.</a></li>
    </ol>
</ul>
