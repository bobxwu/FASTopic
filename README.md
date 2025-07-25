# FASTopic

[![stars](https://img.shields.io/github/stars/bobxwu/FASTopic?logo=github)](https://github.com/BobXWu/Fastopic)
[![PyPI](https://img.shields.io/pypi/v/fastopic)](https://pypi.org/project/fastopic)
[![Downloads](https://static.pepy.tech/badge/fastopic)](https://pepy.tech/project/fastopic)
[![LICENSE](https://img.shields.io/github/license/bobxwu/fastopic)](https://www.apache.org/licenses/LICENSE-2.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17978-<COLOR>.svg)](https://arxiv.org/pdf/2405.17978.pdf)
[![Contributors](https://img.shields.io/github/contributors/bobxwu/fastopic)](https://github.com/bobxwu/fastopic/graphs/contributors/)


[Petr Korab's blog **Topic Modelling in Business Intelligence: FASTopic and BERTopic in Code**](https://towardsdatascience.com/topic-modelling-in-business-intelligence-fastopic-and-bertopic-in-code-2d3949260a37/?sk=9a88660d4e4c64a1d91ad8ede730a520)

[[NeurIPS 2024 paper]](https://arxiv.org/pdf/2405.17978.pdf)
[[Video]](https://recorder-v3.slideslive.com/?share=95127&s=a3c72f9a-4147-4cf0-a7d0-d95e45320df8) 
[[TowardsDataScience Blog]](https://medium.com/@xiaobaowu/easy-fast-and-effective-topic-modeling-for-beginners-with-fastopic-2836781765f0) 
[[Huggingface Blog]](https://huggingface.co/blog/bobxwu/fastopic)  

FASTopic is a fast, adaptive, stable, and transferable topic model, different
from the previous conventional (LDA), VAE-based (ProdLDA, ETM), or clustering-based (Top2Vec, BERTopic) methods.
It leverages optimal transport between the document, topic, and word embeddings from pretrained Transformers to model topics and topic distributions of documents.

If you want to use FASTopic, please cite our [NeurIPS 2024 paper](https://arxiv.org/pdf/2405.17978.pdf) as

```bibtex
@inproceedings{wu2024fastopic,
    title={FASTopic: Pretrained Transformer is a Fast, Adaptive, Stable, and Transferable Topic Model},
    author={Wu, Xiaobao and Nguyen, Thong Thanh and Zhang, Delvin Ce and Wang, William Yang and Luu, Anh Tuan},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```
https://github.com/user-attachments/assets/42fc1f2a-2dc9-49c0-baf2-97b6fd6aea70






- [FASTopic](#fastopic)
  - [Tutorials](#tutorials)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Try FASTopic on your dataset](#try-fastopic-on-your-dataset)
    - [Save and Load](#save-and-load)
    - [Topic info](#topic-info)
    - [Topic hierarchy](#topic-hierarchy)
    - [Topic weights](#topic-weights)
    - [Topic activity over time](#topic-activity-over-time)
  - [APIs](#apis)
    - [Common](#common)
    - [Visualization](#visualization)
  - [Q\&A](#qa)
  - [Contact](#contact)
  - [Related Resources](#related-resources)


## Tutorials

| Tutorial | Link |
| ------ | --- |
| A complete tutorial on FASTopic. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bduHWL5_bvsl4EYOgimCOmU-7RfnXqrX?usp=sharing) |
| FASTopic with other languages. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_b55QpVQGFBX9PsyrYfNxDzbJ1IjrUdH?usp=sharing) |


## Installation

Install FASTopic with `pip`:

```bash
pip install fastopic
```

Otherwise, install FASTopic from the source:

```bash
pip install git+https://github.com/bobxwu/FASTopic.git
```

## Quick Start

Discover topics from 20newsgroups with the topic number as `50`.

```python
from fastopic import FASTopic
from topmost import Preprocess
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

preprocess = Preprocess(vocab_size=10000)

model = FASTopic(50, preprocess)
top_words, doc_topic_dist = model.fit_transform(docs)

```

`top_words` is a list containing the top words of discovered topics.
`doc_topic_dist` is the topic distributions of documents (doc-topic distributions),
a numpy array with shape $N \times K$ (number of documents $N$ and number of topics $K$).


## Usage

### Try FASTopic on your dataset

```python
from fastopic import FASTopic
from topmost.preprocess import Preprocess

# Prepare your dataset.
docs = [
    'doc 1',
    'doc 2', # ...
]

# Preprocess the dataset. This step tokenizes docs, removes stopwords, and sets max vocabulary size, etc.
# preprocess = Preprocess(vocab_size=your_vocab_size, tokenizer=your_tokenizer, stopwords=your_stopwords_set)
preprocess = Preprocess()

model = FASTopic(50, preprocess)
top_words, doc_topic_dist = model.fit_transform(docs)

# (Optional) Plot loss curve for the fitted topic model to monitor convergence
fig = model.plot_loss_arr()
fig.show()
```


### Save and Load

```python

path = "./tmp/fastopic.zip"
model.save(path)

loaded_model = FASTopic.from_pretrained(path)
beta = loaded_model.get_beta()

doc_topic_dist = loaded_model.transform(docs)
# Keep training
loaded_model.fit_transform(docs, epochs=1)
```

### Topic info

We can get the top words and their probabilities of a topic.

```python
model.get_topic(topic_idx=36)

(('cancer', 0.004797671),
 ('monkeypox', 0.0044828397),
 ('certificates', 0.004410268),
 ('redfield', 0.004407463),
 ('administering', 0.0043857736))
```

We can visualize these topic info.

```python
fig = model.visualize_topic(top_n=5)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_info.png?raw=true' with='300pt'></img>


### Topic hierarchy

We use the learned topic embeddings and `scipy.cluster.hierarchy` to build a hierarchy of discovered topics.


```python
fig = model.visualize_topic_hierarchy()
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_hierarchy.png?raw=true' with='300pt'></img>



### Topic weights

We plot the weights of topics in the given dataset.

```python
fig = model.visualize_topic_weights(top_n=20, height=500)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_weight.png?raw=true' with='300pt'></img>




### Topic activity over time

Topic activity refers to the weight of a topic at a time slice.
We additionally input the time slices of documents, `time_slices` to compute and plot topic activity over time.



```python
act = model.topic_activity_over_time(time_slices)
fig = model.visualize_topic_activity(top_n=6, topic_activity=act, time_slices=time_slices)
fig.show()
```

<img src='https://github.com/BobXWu/FASTopic/blob/master/tutorials/topic_activity.png?raw=true' with='300pt'></img>



## APIs

We summarize the frequently used APIs of FASTopic here. It's easier for you to look up.

### Common

| Method | API  | 
|-----------------------|---|
| Fit the model    |  `.fit(docs)` |
| Fit the model and predict documents  |  `.fit_transform(docs)` |
| Predict new documents  |  `.transform(new_docs)` |
| Get topic-word distribution matrix    |  `.get_beta()` |
| Get top words of all topics    |  `.get_top_words()` |
| Get top words and probabilities of a topic    |  `.get_topic(topic_idx=10)` |
| Get topic weights over the input dataset    |  `.get_topic_weights()` |
| Get topic activity over time    |  `.topic_activity_over_time(time_slices)` |
| Save model    |  `.save(path=path)` |
| Load model    |  `.from_pretrained(path=path)` |


### Visualization

| Method | API  | 
|-----------------------|---|
| Visualize topics  | `.visualize_topic(top_n=5)` or `.visualize_topic(topic_idx=[1, 2, 3])`    |
| Visualize topic weights  | `.visualize_topic_weights(top_n=5)` or `.visualize_topic_weights(topic_idx=[1, 2, 3])`    |
| Visualize topic hierarchy  | `.visualize_topic_hierarchy()`    |
| Visualize topic activity  | `.visualize_topic_activity(top_n=5, topic_activity=topic_activity, time_slices=time_slices)`    |




## Q&A

1. **Meet the `out of memory` error. My GPU memory is not enough due to large datasets. What should I do?**

    You can try to set `low_memory=True` and  `low_memory_batch_size` in FASTopic.
    `low_memory_batch_size` should not be too small, otherwise it may damage performance.

    ```python
    model = FASTopic(50, low_memory=True, low_memory_batch_size=2000)
    ```
    Or you can run FASTopic on the CPU as

    ```python
    model = FASTopic(50, device='cpu')
    ```

1. **How can I further accelerate the training?**

    To speed up training, you can increase the learning rate and reduce the number of training epochs. Note that this may slightly affect performance. For example:

    ```python
        model.fit_transform(docs, learning_rate=0.01, epochs=50)
    ```


1. **Can I try FASTopic with the languages other than English?**

    Yes!
    You can pass a multilingual document embedding model, like `paraphrase-multilingual-MiniLM-L12-v2`,
    and the tokenizer and the stop words for your language, like [pipelines of spaCy](https://spacy.io/models).  
    Please refer to the tutorial in Colab.
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_b55QpVQGFBX9PsyrYfNxDzbJ1IjrUdH?usp=sharing)


1. **Loss value can not decline, and the discovered topics all are repetitive.**

    This may be caused by the less discriminative document embeddings,
    due to the used document embedding model or extremely short texts as inputs.  
    Try to
        (1) normalize document embeddings as `FASTopic(50, normalize_embeddings=True)`;
        or
        (2) increase `DT_alpha` to `5.0`, `10.0` or `15.0`, as `FASTopic(num_topics, DT_alpha=10.0)`.

1. **How to reproduce the results? How to get the same results every time?**

    Since FASTopic mainly depends on pytorch and sentence-transformers, you can set random seeds before running FASTopic to get the same results every time:

    ```python
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    model.fit_transform(docs)
    ```
   
1. **Can I use my own document embedding models?**

   Yes! You can wrap your model and pass it to FASTopic:


    ```python
    class YourDocEmbedModel:
        def __init__(self):
            ...

        def encode(self,
                docs: List[str],
                show_progress_bar: bool=False,
                normalize_embeddings: bool=False
            ):
            ...
            return embeddings

    your_model = YourDocEmbedModel()
    model = FASTopic(50, doc_embed_model=your_model)
    ```

1. **Can I use my own preprocess module?**

   Yes! You can wrap your module and pass it to FASTopic:

    ```python
    class YourPreprocess:
        def __init__(self):
            ...

        def preprocess(self, docs: List[str]):
            ...
            train_bow = ...
            vocab = ...

            return {
                "train_bow": train_bow, # sparse matrix
                "vocab": vocab # List[str]
            }

    your_preprocess = YourPreprocess()
    model = FASTopic(50, preprocess=your_preprocess)
    ```

1. **How can I print the running log and progress bars?**

    To enable logging during execution, you can set the verbose parameter to True when initializing the model:

    ```python
    model = FASTopic(num_topics=50, verbose=True)
    ```

## Contact
- We welcome your contributions to this project. Please feel free to submit pull requests.
- If you encounter any issues, please either directly contact **Xiaobao Wu (xiaobao002@e.ntu.edu.sg)** or leave an issue in the GitHub repo.


## Related Resources
- [**TopMost**](https://github.com/bobxwu/topmost): a topic modeling toolkit, including preprocessing, model training, and evaluations.
- [**A Survey on Neural Topic Models: Methods, Applications, and Challenges**](https://github.com/BobXWu/Paper-Neural-Topic-Models)
