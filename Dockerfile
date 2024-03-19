FROM inemo/isanlp

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y liblzma-dev

RUN apt-get install -yqq libffi-dev

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN pyenv install 3.7.17
RUN pyenv global 3.7.17

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install setuptools==41.0.1 numpy scipy scikit-learn==0.22.1 gensim==3.8.1 pandas nltk imbalanced-learn pyyaml grpcio protobuf xgboost

RUN pip install -U git+https://github.com/IINemo/isanlp.git

RUN python -c "import nltk; nltk.download('stopwords')"

COPY src/frustration_classifier /src/frustration_classifier
COPY pipeline_object.py /src/frustration_classifier/pipeline_object.py

### Uncomment this section if embedders are not in the 'models' directory
## fastText embeddings
COPY models /models/

RUN curl -O https://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextskipgram_300_5_2018.tgz
RUN tar -xf araneum_none_fasttextskipgram_300_5_2018.tgz -C models

COPY models /models/
COPY patterns /patterns/

COPY cfg.yaml /cfg.yaml

ENV PYTHONPATH=/src/frustration_classifier/
CMD [ "python", "/start.py", "-m", "pipeline_object"]
