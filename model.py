from torch import nn
from transformers import BertForSequenceClassification

class BertClassifier(BertForSequenceClassification):
    def __init__(self, config):

        # ====================
        #      BERT Setup
        # ====================

        # reulting BERT model is stored in 'self.bert'.
        super().__init__(config)

        self.num_labels = config.num_labels

        # 나중에 config 내부에 해당 field 값을 넣어주면 됨 (영상 참고)
        combined_feat_dim = config.text_feat_dim +  \
                            config.cat_feat_dim +   \
                            config.num_feat_dim

    pass