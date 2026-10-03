"""Render the official chat template without a redundant 155k-token tokenizer."""

import json
from pathlib import Path
from transformers.utils.chat_template_utils import render_jinja_template


class ChatFormatter:
    def __init__(self, directory):
        directory = Path(directory)
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        template_file = directory / "chat_template.jinja"
        self.template = (
            template_file.read_text(encoding="utf-8")
            if template_file.exists()
            else config["chat_template"]
        )
        if isinstance(self.template, list):
            self.template = next(
                value["template"] for value in self.template if value["name"] == "default"
            )
        if isinstance(self.template, dict):
            self.template = self.template["default"]
        self.special = {}
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ):
            value = config.get(key)
            if isinstance(value, dict):
                value = value["content"]
            if value is not None:
                self.special[key] = value

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            raise ValueError(
                "ChatFormatter only renders text; use the existing ExLlama tokenizer for encoding"
            )
        rendered, _ = render_jinja_template(
            conversations=[messages],
            chat_template=self.template,
            **self.special,
            **kwargs,
        )
        return rendered[0]
