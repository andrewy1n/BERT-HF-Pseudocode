from transformers import pipeline, RobertaTokenizerFast

VOCAB_PATH = 'bpe_tokenizer/bpe-bytelevel-vocab.json'
MERGE_PATH = 'bpe_tokenizer/bpe-bytelevel-merges.txt'
tokenizer = RobertaTokenizerFast(vocab_file=VOCAB_PATH, merges_file=MERGE_PATH)

fill_mask = pipeline(
    "fill-mask",
    model="model",
    tokenizer=tokenizer
)

# void <mask>
print(fill_mask("void FUN_080497e1 ( void ) { if ( DAT_0804c10c != '\0' ) { return ; } FUN_08049783 ( ) ; DAT_0804c10c = 1 ; <mask> ; }" ))
