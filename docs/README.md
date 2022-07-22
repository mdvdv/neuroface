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
image = cv2.imread(<image name>)
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

<img src="https://user-images.githubusercontent.com/83948828/180439656-1a44f57c-e38d-49a7-bfcc-9578b0fb9b26.jpg" width="224"/>

```python
# Detect faces on image.
boxes, _ = mtcnn.detect(image)

# Draw detected faces on image.
draw = ImageDraw.Draw(image)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
```

<img src="https://user-images.githubusercontent.com/83948828/180439690-87f57d46-d0a1-4954-8041-9b6a2b374a7f.jpg" width="224"/>

<a name='002'></a>
<h2>2. Face Comparison</h2>

In progress.

<a name='003'></a>
<h2>3. Facial Landmark Detection</h2>

In progress.

<a name='004'></a>
<h2>4. Pose Landmark Detection</h2>

In progress.

<a name='005'></a>
<h2>5. Facial Expression Recognition</h2>

In progress.
