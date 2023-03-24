# How to use our public age and gender model

An introduction to our model for 
age and gender prediction based on
[wav2vec 2.0](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/).
The model is available from 
[doi:10.5281/zenodo.7761387](https://doi.org/10.5281/zenodo.7761387)
and released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The model was created
by fine-tuning the pre-trained
[wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust)
model on
[aGender](https://paperswithcode.com/dataset/agender), 
[Mozilla Common Voice](https://commonvoice.mozilla.org/), 
[Timit](https://catalog.ldc.upenn.edu/LDC93s1) and 
[Voxceleb 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
We provide two models: 
one with all 24 transformer layers and 
a stripped-down version with six transformer layers.
The models were exported to
[ONNX](https://onnx.ai/).
Further details are given in the associated 
paper (tba)
and [notebook](./notebook.ipynb).

## Quick start

Create / activate Python virtual environment and install 
[audonnx](https://github.com/audeering/audonnx).

```
$ pip install audonnx
```

Load the model with six layers and test on random signal.

```python
import audeer
import audonnx
import numpy as np


url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

sampling_rate = 16000
signal = np.random.normal(size=sampling_rate).astype(np.float32)
model(signal, sampling_rate)
```
```
{'hidden_states': array([[ 0.02783544,  0.01402022,  0.03839185, ...,  0.00786646,
         -0.09332313,  0.0915948 ]], dtype=float32),
 'logits_age': array([[0.3961048]], dtype=float32),
 'logits_gender': array([[ 0.32810774, -0.56528044,  0.0317882 ]], dtype=float32)}
```

The 'hidden_states' are the pooled states of the last transformer layer, 
'logits_age' provides scores for age in a range of approximately 0...1 (== 100 years) 
and 'logits_gender' expresses the confidence for being female, male or child.

## Tutorial

For a detailed introduction, please check out the [notebook](./notebook.ipynb).

```bash
$ pip install -r requirements.txt
$ jupyter notebook notebook.ipynb 
```

## Citation

If you use our model in your own work, please cite the following
paper (tba)
