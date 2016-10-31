## What is grammar?

If English is your native language you probably learned English [grammar](https://en.wikipedia.org/wiki/Category:English_grammar) rules and terminology like [this](https://en.wikisource.org/wiki/The_Grammar_of_English_Grammars/Part_II):

### Parts of Speech

- [Adjectives](https://en.wikipedia.org/wiki/Adjective)
- Adverbs
- Articles
- Conjunctions
- Interjections
- Nouns
- Prepositions
- Pronouns
- Verbs

### Punctuation

- Comma
- Colon
- Semicolon
- Apostrophe
- Quotation Marks
- Dash
- Hyphen
- End of sentence (e.g. period, quesiton mark, exclamation mark)
- Other punctuation

### Mechanics

- Spelling
- Capitalization
- Abbreviation
- Numbers
- Italics
- Underlining
- Phrasal Verbs
- Idioms
- Compound Words

### Sentence Structure and Style

- Comparison
- Conditional Sentences
- Qualifiers and Quantifiers
- Mixed Constructions
- Negatives
- Modifiers
- Parallelism
- Shifts in Writing
- Sentence style
- Transitions and Transitional Devices

However, in computer science or NLP we ignore most of these grammar rules and usually mean Chomsky's definition of "grammar". For Chomsky, a gramatical (think "gramatically correct") sentence is any sentence which a native speaker can easily:

- "recognize as intuitively acceptable"
- utter with the same intonation that most others would utter it
- remember

That last one "remember", probably seems a little fuzzy, subjective, and you're right. But the fuzziness is overshadowed by the significant increase in memorizability of a grammatical sentence relative to an ungramatical sentence of the same length. This was at the core of Chomsky's revelation (at the age of 28) that semantics and grammar could be cleanly delineated and studied separately. Chomsky's 100 page treatise on the subject included the now popular example of a grammatical but nonsensical sentence.

> Colorless green ideas sleep furiously.

If you read this phrase as poetry, rather than grammatical English prose, it gives the impression of an idea plaguing the speaker with confusion, contradiction, and passionate internal disagreement, but there is no literal meaning you could derive from this sentence. It is a valid, gramatical English sentence. It is not semantically meaningful when taken literally, objectively. Only poetically, subjectively does it give an impression of meaning, but that meaning will vary from reader to reader to a much greater degree than most "normal", semantically valid, sentences. We can only wonder if this example phrase was sparked by Chomsky's confused state of mind as he contemplated the separation of grammar from meaning during many a sleepless night.

If you've ever worked with Regular Expressions, you may think of the "acceptable" part as being a valid `re.match()` by a regular expression that encompasses all the rules of english grammar and all the correct spellings of all words in English. Others of you that have worked with Finite State Machines will recognize this as a "formal grammar". But this formal grammar is constantly evolving and is shaped by various subgroups of english speakers, so is more inclusive and malleable than software language syntax and formal grammars.

The Grammar, or syntactic structure of a sentence is much easier for a machine to recognize than is the semantic structure. Only recently have statistical methods applied to vast corpora succeeded at approximating the semantic as well as grammatical structure with a single neural network (SyntaxNet by Google).
