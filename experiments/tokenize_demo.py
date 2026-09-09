"""What `tokenizer.encode` actually hands back, for the running example.

The chapter says the text becomes numbers. This prints the numbers.
"""

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode("Hi my name is")

print(encoded)
for i in encoded:
    print(f"{i:>6}  {tokenizer.decode([i])!r}")
