Leveraging Ontological Schema Information in Embedding Models for Knowledge Graphs
==================================================================================

1. Overview
-----------------------------------------------------------------

This package contains all source code and datasets for:
1.  Efficient training of Energy-Based Models for Knowledge Graphs.
2.  Leveraging Schema Information (expressed in RDF Schema) during learning.

The models implemented in this package extend the models proposed in the following works:
- **Translating Embeddings** (TransE) - Bordes et al., NIPS 2013.
- **Semantic Matching Energy** (SME) - Bordes et al., MLJ 2014.
- **Structured Embeddings** (SE) - Bordes et al., AAAI 2011.

Content of the package:
- learn_parameters.py : handles the main learning process.
- data/ : contains all datasets, namely FB13, FB15k, WN11, WN18, DBpedia 2014, YAGO3 - FB* and WN* are used in referenced works, while DBpedia 2014 and YAGO3 are two additional large datasets from the "Semantic Web" literature, enriched with RDF Schema information.
- energy/ : contains all code neede for defining and training the models (loss functionals, dissimilarity functions, adaptive learning rate selectors such as AdaGrad, AdaDelta and Momentum, and so on and so forth)
- scripts/ : contains all scripts used for running the experiments - each script generates a sequence of commands, and multiple commands can be run in parallel (e.g. by using the GNU Parallel utility).


2. 3rd Party Libraries
-----------------------------------------------------------------

This package is written in Python, and uses the Theano library for automatic symbolic differentiation, and efficiently compiling the energy functions and their gradients on heterogeneous architectures (e.g. CPU, GPU). It also uses Numpy, Scipy, PyMongo (for saving the learned models and experimental results on MongoDB), and Seaborn (for generating plots).

3. References
-----------------------------------------------------------------
- A. Bordes and E. Gabrilovich, “Constructing and mining web-scale knowledge graphs” - Tutorial at KDD 2014 and WWW 2015
- A. Bordes et al. “A semantic matching energy function for learning with multi-relational data - application to word-sense disambiguation” - MLJ 2014
- A. Bordes et al. “Translating embeddings for modeling multi-relational data” - NIPS 2013
- A. Bordes et al. “Learning structured embeddings of knowledge bases” - AAAI 2011
