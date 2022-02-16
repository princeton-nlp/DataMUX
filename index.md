---
layout: default
---

\[[Code](https://github.com/princeton-nlp/DataMUX)\]

## Abstract
In this work, we show that deep neural networks are capable of generating accurate predictions over *mixtures* of inputs. We introduce data multiplexing (DataMUX), a novel technique that enables networks to process multiple inputs simultaneously using a single compact representation, resulting in increased throughput with minimal extra space requirements. Our approach uses two key components -- a *multiplexing* layer that performs a fixed linear transformation to each input before combining them to create a single `mixed' representation which is processed by the base network, and a *demultiplexing* layer that converts the network's output back into independent representations before producing predictions for each input. We demonstrate the viability of DataMUX for multiple architectures (Transformers, and to a lesser extent MLPs and CNNs) across six different tasks spanning sentence classification, named entity recognition and image classification. For instance, DataMUX for Transformers can multiplex up to 20x/40x, achieving 11x/18x throughput with minimal absolute performance drops of <2% and <4%, respectively over a standard Transformer on MNLI, a natural language inference task.We also provide a theoretical construction for multiplexing in self-attention networks and analyze the effect of various design elements in DataMUX.

## Authors
<div class="container">
    <figure>
    <a href="https://vishvakmurahari.com"><img src="assets/photos/vishvak_photo.jpg" width="110" height="110" alt="" class="profphoto" id="firstprofphoto"></a>
        <figcaption><a href="https://vishvakmurahari.com">Vishvak Murahari</a></figcaption>
    </figure>
    <figure>
    <a href="http://carlosejimenez.com/"><img src="assets/photos/carlos-photo.jpeg" width="110" height="110" alt="" class="profphoto"></a>
        <figcaption><a href="http://carlosejimenez.com/">Carlos E. Jimenez</a></figcaption>
    </figure>
    <figure>
    <a href="https://runzhe-yang.science"><img src="assets/photos/runzhe-photo.jpg" width="110" height="110" alt="" class="profphoto"></a>
        <figcaption><a href="https://runzhe-yang.science">Runzhe Yang</a></figcaption>
    </figure>
    <figure>
    <a href="https://www.cs.princeton.edu/~karthikn/"><img src="assets/photos/karthik-photo.jpeg" width="110" height="110" alt="" class="profphoto"></a>
        <figcaption><a href="https://www.cs.princeton.edu/~karthikn/">Karthik Narasimhan</a></figcaption>
    </figure>
</div>

### Citation

```
@article{datamux
  title={DataMUX: Data Multiplexing for Neural Networks,
  author={Vishvak Murahari and Carlos E. Jimenez and Runzhe Yang and Karthik Narasimhan},
  journal={arXiv preprint arXiv:TODO},
  year={2022},
}
```

