![Typing SVG](https://readme-typing-svg.herokuapp.com?size=100&color=BBBBBB&center=true&vCenter=true&width=1875&height=100&lines=NeuroFace)

---

<a name='000'></a>
<h2>Table of Contents</h2>

<ul>
    <ol type='1'>
        <li><a href='#001'>Face Detection</a></li>
        <li><a href='#002'>Face Comparison</a></li>
        <li><a href='#003'>Facial Landmark Detection</a></li>
        <li><a href='#004'>Pose Landmark Detection</a></li>
	<li><a href='#005'>Facial Expression Recognition</a></li>
    </ol>
</ul>

<a name='001'></a>
<h2>1. Face Detection</h2>

Face detection on raw images using [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf) on GPU device.

```python
import cv2
import torch
from PIL import Image, ImageDraw

from neuroface import MTCNN

# Initialize GPU device if available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN on GPU device.
mtcnn = MTCNN(keep_all=True, device=device).eval()

# Upload image and change color space.
image = cv2.imread(<select image>)
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

<img src="https://user-images.githubusercontent.com/83948828/180439656-1a44f57c-e38d-49a7-bfcc-9578b0fb9b26.jpg" width="224"/>

```python
# Detect face boxes on image.
boxes, _ = mtcnn.detect(image)

# Draw detected faces on image.
draw = ImageDraw.Draw(image)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
```

<img src="https://user-images.githubusercontent.com/83948828/180439690-87f57d46-d0a1-4954-8041-9b6a2b374a7f.jpg" width="224"/>

<a name='002'></a>
<h2>2. Face Comparison</h2>

Building face embeddings using [InceptionResnetV1](https://arxiv.org/pdf/1503.03832.pdf) pretrained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf) dataset.

```python
import torch
import torchvision.io as io

from neuroface import MTCNN, InceptionResnetV1
from neuroface.face.comparison.distance import distance

# Initialize GPU device if available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and InceptionResnetV1 on GPU device.
mtcnn = MTCNN(keep_all=True, device=device).eval()
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Upload images and rearrange dimensions (C H W --> H W C).
image = io.read_image(<select image>).to(device).permute(1, 2, 0)

# Detect faces on images.
face = mtcnn(image)
```

Calculating distance between obtained embeddings.

- `0` to select Euclidian distance:

$$d(p, q)=\sqrt{\sum_{i=1}^{n} (p_i-q_i)^2}.$$

- `1` to select Euclidian distance with L2 normalization:

$$|p|=\sqrt{\sum_{i=1}^{n} |p_i|^2}, |q|=\sqrt{\sum_{i=1}^{n} |q_i|^2}$$

$$d(p, q)=\sqrt{\sum_{i=1}^{n} (|p|_i-|q|_i)^2}.$$

- `2` to select cosine similarity.

$$d(p, q)=\frac{\sum_{i=1}^{n} (p_{i}*q_{i})}{\sqrt{\sum_{i=1}^{n} (p_{i}^2)}*\sqrt{\sum_{i=1}^{n} (q_{i}^2})}.$$

- `3` to select Manhattan distance:

$$d(p, q)=\sum_{i=1}^{n} |p_i-q_i|.$$

```python
# Rearrange dimensions (B H W C --> B C H W) and build face embeddings.
embedding = resnet(face.permute(0, 3, 1, 2))

# Calculate distance between embeddings.
print(distance(<select embedding>, <select embedding>, distance_metric=<select metric>))
```

<table>
  <thead>
    <tr>
      <th></th>
      <th><img src="https://user-images.githubusercontent.com/83948828/180442835-8992db0d-50ba-4a11-be2d-ad1af018d653.jpg" width="100" height="100"/></th>
      <th><img src="https://user-images.githubusercontent.com/83948828/180442752-e4dea08d-afbd-496d-bf7b-d1f68ea902ab.jpg" width="100" height="100"/></th>
      <th><img src="https://user-images.githubusercontent.com/83948828/180442888-33c0d4a0-4c9e-44b9-b9fa-46e3c219160d.jpg" width="100" height="100"/></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Euclidian Distance</td>
      <td>0.0</td>
      <td>1.9461</td>
      <td>0.5072</td>
    </tr>
    <tr>
      <td>Euclidian Distance with L2 Normalization</td>
      <td>0.0</td>
      <td>1.9461</td>
      <td>0.5072</td>
    </tr>
    <tr>
      <td>Cosine Similarity</td>
      <td>0.0</td>
      <td>0.4914</td>
      <td>0.2318</td>
    </tr>
    <tr>
      <td>Manhattan Distance</td>
      <td>0.0</td>
      <td>25.2447</td>
      <td>12.7753</td>
    </tr>
  </tbody>
</table>

<a name='003'></a>
<h2>3. Facial Landmark Detection</h2>

```python
import cv2

from neuroface import FaceMesh

# Initialize FaceMesh.
model = FaceMesh(static_image_mode=True, max_num_faces=1)

# Upload image.
image = cv2.imread(<select image>)

# Detect facial landmarks.
face_array = model.detect(image)
```

<a name='004'></a>
<h2>4. Pose Landmark Detection</h2>

In progress.

<a name='005'></a>
<h2>5. Facial Expression Recognition</h2>

In progress.
