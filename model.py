import torch
from torch import nn
from transformers import BertForSequenceClassification
from classifiers.mlp import MLPClassifier

class BertWithMlp(BertForSequenceClassification):
    def __init__(
        self,
        config,
        input_dim   = None,
        hidden_dim  = None,
        num_classes = 2,
        dropout     = 0.1
        ):

        # ====================
        #      BERT Setup
        # ====================

        # resulting BERT model is stored in 'self.bert'.
        super().__init__(config)

        self.num_labels     = config.num_labels
        combined_feat_dim   = config.text_feat_dim + config.cat_feat_dim + config.num_feat_dim

        # ===================
        #      MLP Setup
        # ===================
        self.mlp = MLPClassifier(
            combined_feat_dim,
            None,
            hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        print(" =========== mlp model =========== ")
        print(self.mlp)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.bn      = nn.BatchNorm1d(config.num_feat_dim)

    def forward(
        self,
        input_ids       = None,
        attention_mask  = None,
        token_type_ids  = None,
        position_ids    = None,
        head_mask       = None,
        inputs_embeds   = None,
        labels          = None,
        output_attentions = None,
        cat_input       = None,
        num_input       = None
    ):
        # ====================
        #     BERT forward
        # ====================
        #TODO: 더 많은 인자 추가해주기
        logits = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        # TODO: cls 검증 필요
        cls = logits[1]
        # Apply dropout to cls
        cls = self.dropout(cls)
        # Apply batch normalization to numerical features
        #FIXME: Batch norm 오류 남
        # num_input = self.bn(num_input)

        # ====================
        #      MLP forward
        # ====================
        cat_input = cat_input.view(-1, 1)
        num_input = num_input.view(-1, 1)
        all_feats = torch.cat((cls, cat_input, num_input), dim=1)
        output = self.mlp(all_feats)

        return output
