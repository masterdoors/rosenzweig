import numpy

import json   
from scipy.spatial.distance import cosine


#compare morphology features
class Vertex:
    def __init__(self,rule,lemmas,directs,nots,sem_pred,sem_arg,vectorizer):
        self.rule = rule
        self.lemmas = lemmas
        self.pred = sem_pred
        self.arg = sem_arg
        self.directs = directs
        self.nots = nots
        self.vectorizer = vectorizer
        
    def vectComp(self,lemma,lemmas,vectors,all_lemmas):    
        if lemma in vectors:
            for l in lemmas:
                if l in vectors[lemma]:
                    dist = vectors[lemma][l]
                    if dist < 0.39:
                        return True
                else:
                    if l in all_lemmas:
                        l_v = all_lemmas[l]
                    else:  
                        l_v = self.vectorizer.transform(l)
                        all_lemmas[l] = l_v
                    if lemma in all_lemmas: 
                        lemma_v = all_lemmas[lemma]
                    else:
                        lemma_v = self.vectorizer.transform(lemma)
                    dist = abs(cosine(lemma_v,l_v))
                    vectors[lemma][l] = dist
                    if dist < 0.39:
                        return True
        else:
            if lemma in all_lemmas:
                lemma_v = all_lemmas[lemma]
            else:
                lemma_v = self.vectorizer.transform(lemma)  
                all_lemmas[lemma] = lemma_v

            vectors[lemma] = {}
            for l in lemmas:
                if l in all_lemmas:
                    l_v = all_lemmas[l]
                else:  
                    l_v = self.vectorizer.transform(l)
                    all_lemmas[l] = l_v

                dist = abs(cosine(lemma_v,l_v))
                vectors[lemma][l] = dist
                if dist < 0.39:
                    return True

        return False    
        
    def getCommonPart(self,rule,clause):
        if len(rule) > 0:
            r = []
            for k in rule:
                if k in clause:
                    r.append((rule[k] == clause[k]))
                else:
                    r.append(False)             
            
            res = 0
            if len(r) > 0:
                res = all(r)
                
            return float(res)
        else:
            return 1.0  
        
    def getSemSim(self,token):
        pred_sim = 1.0
        if self.pred:
            pred_sim = int("sem_pred" in token and token["sem_pred"] == 1)
        pred_arg = 1.0
        
        if self.arg:
            pred_arg = int ("sem_role" in token and token["sem_role"] == self.arg)  
            
        return pred_sim * pred_arg                     
        
    def compare(self,token,vectors, all_lemmas):
        lemma_sim = 1
        if len(self.nots) > 0:
            if self.vectComp(token["lemma"],self.nots,vectors,all_lemmas):
                lemma_sim = 0.
        else:            
            if len(self.lemmas) > 0:
                lemma_sim = float(self.vectComp(token["lemma"],self.lemmas,vectors,all_lemmas))    
            
        return self.getSemSim(token)*self.getCommonPart(self.rule,token["morph"]) * lemma_sim

class Chain:
    def __init__(self,vertex,parent,plink_type,text):
        self.vertex = vertex
        self.parent = parent
        self.parent_link_type = plink_type
        self.text = text
    
    def compare(self,token, vectors, all_lemmas):
        
        c = self.vertex.compare(token, vectors, all_lemmas)

        if token["syn_parent"]:  
            if self.parent:
                link_weight = 1.0
                if self.parent_link_type:
                    link_weight = float(self.parent_link_type == token["syn_name"])

                #process "*"
                if len(self.parent.vertex.rule) == 0 and len(self.parent.vertex.lemmas) == 0 and not self.parent.vertex.arg and not self.parent.vertex.pred:
                    target_rule = self.parent.parent
                
                    if target_rule:
                        cur_token =  token["syn_parent"]
                        res = target_rule.compare(cur_token, vectors, all_lemmas)
                        while cur_token and res == 0:
                            cur_token = cur_token["syn_parent"] 
                            
                            if cur_token:
                                res = target_rule.compare(cur_token,vectors, all_lemmas)
                                
                        return c * res    
                    else:
                        return c
                if c > 0: 
                    return c * self.parent.compare(token["syn_parent"],vectors,all_lemmas)
                else:
                    return 0
            else:
                return c #* self.compare(token["syn_parent"])
        else:
            if self.parent:
                return 0
            else:
                return c

class Rule:
    def __init__(self,dicts,lemm_vect):  
        self.dicts = dicts
        self.vectorizer = lemm_vect
        self.all_lemmas = {}
        
    def from_struct(self,struct):
        self.chains = []
        
        for rule in struct:
            parent = None
            chain = None
            for c in reversed(rule):
                lemmas = set()
                directs = set()
                nots = set()
                rule_ = {}  
                is_pred = False
                arg_id = ""
                
                if "not" in c:
                    for dname in c["not"]:
                        nots = nots.union(set([dname]))
                if "direct" in c:
                    for dname in c["direct"]:
                        if dname.isupper():
                            directs = directs.union(self.dicts[dname])
                        else:
                            directs = directs.union(set([dname]))                        
                else:
                    #if there is a direct, then we ignore other fields    
                    if "lexis" in c:
                        for dname in c["lexis"]:
                            if dname.isupper():
                                lemmas = lemmas.union(self.dicts[dname]) 
                            else:
                                lemmas = lemmas.union(set([dname]))    
  
                    if "morph"  in c:    
                        rule_ = c["morph"]
    

                    if "sem_pred" in c:
                        is_pred  = c["sem_pred"]
                        
                    if "sem_role" in c:
                        arg_id = c["sem_role"]                    
                    
                lemmas_d =  self.vectorizer.transform(lemmas)
                nots_d = self.vectorizer.transform(nots)

                for l in lemmas_d:
                    self.all_lemmas[l] = lemmas_d[l]
                for l in nots_d: 
                    self.all_lemmas[d] = nots_d[l]

                vertex = Vertex(rule_,lemmas,directs, nots,is_pred,arg_id,self.vectorizer)
                plink_type = None
                if "syn_name" in c:
                    plink_type = c["syn_name"]
                    
                chain = Chain(vertex,parent,plink_type,str(rule))
                parent = chain
                
            self.chains.append(chain)          
        
    
    def from_json(self,js_str):
        dct_form = json.loads(js_str)
        self.from_struct(dct_form) 
        
    def compare(self,clause,txt,verbose=False,vectors={}):

        res = []
        for c in self.chains:
            if (not c.parent) and len(c.vertex.directs) > 0:
                found = False
                for phrase in c.vertex.directs:
                    f_idx = txt.find(phrase)                    
                    c1 = False
                    c2 = False
                    if f_idx > -1:
                        if f_idx == 0:   
                            c1 = True
                        else:
                            if not txt[f_idx-1].isalpha():
                                c1 = True
                                
                        if f_idx+len(phrase) == len(txt):
                            c2 = True
                        else:  
                            if not txt[f_idx+len(phrase)].isalpha(): 
                                c2 = True
                                
                        if c1 and c2:
                            found = True                     
                                    
                if found:
                    res.append(1)
                else:
                    res.append(0)            
  
            else:    
                item = c.compare(clause, vectors, self.all_lemmas)
                res.append(item)    
        return numpy.asarray((res))

       
