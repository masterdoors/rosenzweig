# coding: utf-8

import os
import json
import numpy
import re
import yaml
import gensim

from isanlp.processor_remote import ProcessorRemote
from isanlp import PipelineCommon
from isanlp.annotation_repr import CSentence
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd

from pattern_matcher.patterns_matcher import Rule
import time


class Vectorizer:
    def __init__(self,path):
        assert os.path.exists(path), 'Pattern matcher: fasttext model is not found.'
        self.model = gensim.models.KeyedVectors.load(path)
        
    def transform(self,lemma=[]):    
        if isinstance(lemma, list) or isinstance(lemma, set):
            res = {}
            for l in lemma:
                res[l]  = self.model[l]
            return res
        else:
            return self.model[lemma]

class Processor:
    
    def objectifyClauses(self, text,lemma,postag,morph,syntax_dep_tree,srl__):
        #make common list of objects instead of separate lists for features
        outp = []
        cl = [] 
        for i in range(len(postag)):
            srl = srl__[i]

            aggregated = zip(morph[i],lemma[i],syntax_dep_tree[i])

            cll = []
            for l in aggregated:
                morph, lemma, synt = l
                node = {}
                node["morph"] = morph
                node["syn_parent"] = synt.parent
                node["syn_name"] = synt.link_name
                node["lemma"] = lemma #self.vectorizer.transform(lemma)

                cll.append(node)

            for srl_ in srl:
                for pi in range(srl_.pred[0],srl_.pred[1]+1):
                    cll[pi]["sem_pred"] = True

                for a in srl_.args:
                    for ai in range(a.begin,a.end+1):
                        cll[ai]["sem_arg"] = a.tag   
                            
            cl.append(cll)            
        outp.append(cl)  
            
        #make tree
        trees = []
        for cls in outp:
            trees_parts = []
            for cl_ in cls:
                tp = []
                for cl in cl_:
                    if cl["syn_parent"] is not None:
                        if cl["syn_parent"] > -1:
                            cl["syn_parent"] = cl_[cl["syn_parent"]]
                        else:
                            cl["syn_parent"] = None
                    tp.append(cl)    
                trees_parts.append(tp)
            trees.append(trees_parts)        
        return trees
    
    def load_dicts(self,path):
        self.dicts = {}
        for f in os.listdir(path):
            dict_name = os.path.join(path, f)
            with open(dict_name,'r') as df:
                dict_strs = df.read().strip().split("\n")
                self.dicts[f] = set(dict_strs)
                
    def __init__(self,patterns_folder, ling_cfg, logger,verbose = False):
            
        self.verbose = verbose
        self.logger = logger          

        assert os.path.exists(patterns_folder), 'Pattern matcher: pattern folder is not found.'
        self.load_dicts(os.path.join(patterns_folder,"Dicts"))
        self.cat2rule = {}
        self.vectorizer = Vectorizer(ling_cfg['fastext_model'])
        
        for filename in os.listdir(patterns_folder):
            if filename.endswith(".json"):     
                with open(os.path.join(patterns_folder,filename), "r") as read_file:    
                    pattern_file = json.load(read_file)
                    
                for k in pattern_file:
                    self.cat2rule[k] = Rule(self.dicts,self.vectorizer)
                    self.cat2rule[k].from_struct(pattern_file[k])
                
                    if verbose:
                        logger.info ("Pattern " + str(k) + " was successully loaded " +  str(len(self.cat2rule[k].chains)))
        
    def process(self,text,lemma,postag,morph,syntax_dep_tree,srl):
        lin_objects = self.objectifyClauses(text,lemma,postag,morph,syntax_dep_tree,srl)    
        text_ = text.lower()
        i = 0
        vectors = {}
        rule_rept = []
        
        for clause in lin_objects:
            features_list = []
            for k in self.cat2rule:
                total_score = numpy.zeros((len(self.cat2rule[k].chains),))
                for clause_part in clause:
                    for tree_sample in clause_part:
                        res = self.cat2rule[k].compare(tree_sample,text_,vectors)
                        total_score += res
    
                for t in total_score:
                    features_list.append(t)  
                    
            rule_rept.append(features_list)
            i += 1
                    
        return numpy.asarray(rule_rept) 
