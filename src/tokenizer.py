import torch
import tiktoken



class TiktokenTokenizer():

    def __init__(self, name) -> None:
        self.enc = tiktoken.get_encoding(name)
        self.encode = lambda s: self.enc.encode(
            s, allowed_special={"<|endoftext|>"})
        self.pad_token = self.enc.eot_token

    def __call__(self,
                 text,
                 max_length=None,
                 padding=None,
                 truncation=False,
                 return_tensors=None):
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            mask += [0] * (max_length - len(ids))
            ids += [self.pad_token] * (max_length - len(ids))

        if return_tensors == "pt":
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask)

        return {"input_ids": ids, "attention_mask": mask}


