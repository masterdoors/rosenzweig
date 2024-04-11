from pattern_matcher.processor import Processor

import pickle
import yaml
import numpy as np
import logging

logging.basicConfig(filename="/sample.log", level=logging.INFO)
logger = logging.getLogger('rosenzweig')
 
class ProcessorRosenzweig:
    def __init__(self, config_file):
        try: 
            cfg_file = open(config_file)
            self._cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
            self._clf_model = pickle.load(open(self._cfg['clf_model_path'], 'rb'))
            self._proc = Processor(self._cfg['patterns_path'], self._cfg, logger,verbose = False)
        except Exception as e:
            logger.error("Cannot initialize pattern matcher:" + str(e))  

    def __call__(self, text,tokens,sentences, postag, morph, lemma, syntax_dep_tree,srl):
        assert self._clf_model
        assert self._proc
        res = []
        for i,sent in enumerate(sentences):
            sbegin = sent.begin
            send = sent.end - 1
            main = text[tokens[sbegin].begin:tokens[send].end]
            context = text[:tokens[sbegin].begin] + text[tokens[send].end:]
            frag_patterns = self._proc.process(main,[lemma[i]],[postag[i]],[morph[i]],[syntax_dep_tree[i]],[srl[i]])    
            context_patterns = self._proc.process(context,[lemma[i]],[postag[i]],[morph[i]],[syntax_dep_tree[i]],[srl[i]])    
            X_f = frag_patterns
            X_c = context_patterns
            Xpat = np.hstack([X_f,X_c]) 
            proba = self._clf_model.predict_proba(Xpat)
            res.append(str({self._cfg['class_alias'][j]:p for j,p in enumerate(proba[0])}))

        return res






