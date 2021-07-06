from isanlp import PipelineCommon
from processor_rosenzweig import ProcessorRosenzweig

def create_pipeline(delay_init):
    pipeline_default = PipelineCommon([(ProcessorRosenzweig(config_file='/cfg.yaml'),
                                        ['text', 'tokens', 'sentences', 'lemma', 'morph', 'postag', 'syntax_dep_tree','srl'],
                                        {0: 'rosenzweig'})
                                       ],
                                      name='default')

    return pipeline_default

