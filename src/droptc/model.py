
import torch
import torch.nn as nn

class DroPTC(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=7, freeze_embedding=False):
        """
        Args:
            embedding_model: A Hugging Face-compatible transformer model with a .forward method
            tokenizer: The corresponding tokenizer
            hidden_dim: Size of the hidden layer in the classifier
            dropout_rate: Dropout probability
            num_class: Number of classes in the output
            freeze_embedding: Whether to freeze the embedding's parameters
        """
        super(DroPTC, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )
        self.hidden = None
        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        hidden = self.dense(pooled)
        self.hidden = hidden
        logits = self.classifier(hidden)  # [batch_size, num_classes]
        return logits
    

class DroneLog(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=7, freeze_embedding=False):
        super(DroneLog, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.encoder = nn.LSTM(
            input_size=self.embedding_model.config.hidden_size,
            hidden_size=384,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )
        self.hidden = None
        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        last_hidden_state, _ = self.encoder(last_hidden_state)
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        hidden = self.dense(pooled)
        self.hidden = hidden
        logits = self.classifier(hidden)  # [batch_size, num_classes]
        return logits
    

class NeuralLog(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=7, freeze_embedding=False):
        super(NeuralLog, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_model.config.hidden_size,
                    nhead=12,
                    dim_feedforward=2048,
                    batch_first=True,
                    dropout=0.1,
                ),
                num_layers=1,
            )
        self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )
        self.hidden = None
        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        last_hidden_state = self.encoder(last_hidden_state)
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        hidden = self.dense(pooled)
        self.hidden = hidden
        logits = self.classifier(hidden)  # [batch_size, num_classes]
        return logits
    

class TransSentLog(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=7, freeze_embedding=False):
        super(TransSentLog, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_model.config.hidden_size,
                    nhead=2,
                    dim_feedforward=2048,
                    batch_first=True,
                    dropout=0.1,
                ),
                num_layers=2,
            )
        self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )
        self.hidden = None
        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        last_hidden_state = self.encoder(last_hidden_state)
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        hidden = self.dense(pooled)
        self.hidden = hidden
        logits = self.classifier(hidden)  # [batch_size, num_classes]
        return logits
    
class DroLoVe(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=7, freeze_embedding=False):
        super(DroLoVe, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_model.config.hidden_size,
                    nhead=4,
                    dim_feedforward=2048,
                    batch_first=True,
                    dropout=0.1,
                ),
                num_layers=3,
            )
        self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )
        self.hidden = None
        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        last_hidden_state = self.encoder(last_hidden_state)
        pooled = last_hidden_state[:, 0, :]  # CLS pooling [batch_size, hidden_dim]
        hidden = self.dense(pooled)
        self.hidden = hidden
        logits = self.classifier(hidden)  # [batch_size, num_classes]
        return logits
