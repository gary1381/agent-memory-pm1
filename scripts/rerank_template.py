"""Write a copy of a Qwen3-Reranker GGUF whose rerank template carries our own task instruction.

llama.cpp reads the reranker prompt from `tokenizer.chat_template.rerank` and only substitutes {query} and
{document}; the instruction ("Given a web search query ...") is baked into the file. Qwen recommends a
task-specific English instruction (+1-5%), so we bake ours into a sibling file.

  .venv/bin/python scripts/rerank_template.py in.gguf out.gguf "<instruction>"
"""
import sys

from gguf import GGUFReader, GGUFValueType, GGUFWriter
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

KEY = 'tokenizer.chat_template.rerank'
DEFAULT = 'Given a web search query, retrieve relevant passages that answer the query'

src, dst, instruction = sys.argv[1:4]
reader = GGUFReader(src)
field = reader.fields[KEY]
template = bytes(field.parts[field.data[0]]).decode()
assert DEFAULT in template, 'unexpected template'
arch = bytes(reader.fields['general.architecture'].parts[reader.fields['general.architecture'].data[0]]).decode()
writer = GGUFWriter(dst, arch=arch, endianess=reader.endianess)
copy_with_new_metadata(reader, writer,
                       {KEY: MetadataDetails(GGUFValueType.STRING, template.replace(DEFAULT, instruction))}, [])
print('wrote', dst)
