#I @"C:\Users\cybernetic\.nuget\packages\"
#I @"C:\Users\cybernetic\source\repos\"
#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#time "on"

open System.Collections.Generic
open System
open Prelude.Common
open Newtonsoft.Json.Linq

(*def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs*)

let get_pairs (word: string []) =
    // Return set of symbol pairs in a word.
    // Word is represented as tuple of symbols (symbols being variable-length strings).
    let pairs = HashSet()
    let mutable prev_char = string word.[0]

    for str in word.[1..] do
        pairs.Add((prev_char, str)) |> ignore
        prev_char <- str

    pairs

get_pairs ("farimflimflam".ToStringArray())
|> Seq.toArray // = [|('f', 'a'); ('a', 'r'); ('r', 'i'); ('i', 'm'); ('m', 'f'); ('f', 'l'); ('l', 'i'); ('i', 'm'); ('m', 'f'); ('f', 'l'); ('l', 'a'); ('a', 'm')|]

//
let jsontxt =
    IO.File.ReadAllText @"D:\Downloads\NeuralNets\galactica\tokenizer.json"

//type for huggingface gpt2 tokenizer.json

(*{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "<s>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
  ],
  "normalizer": {
    "type": "NFKC"
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "String": "SPL1T-TH1S-Pl3A5E"
        },
        "behavior": "Removed",
        "invert": false
      },
      {
        "type": "Digits",
        "individual_digits": true
      }
    ]
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "vocab": {
      "<s>": 0,
      "<pad>": 1,
      "</s>": 2,
      "<unk>": 3
    },
    "merges": [
      "Ġ t",
      "i n",
      "Ġ a"
    ]
  }
}*)

type ModelTokenizer =
    { Type: string
      Dropout: string
      Unk_token: string
      Continuing_subword_prefix: string
      End_of_word_suffix: string
      Fuse_unk: bool
      Vocab: Dictionary<string, int>
      Merges: string [] }

type Tokenizer =
    { Version: string
      Truncation: string
      Padding: string
      Added_tokens: JObject []
      Normalizer: Dictionary<string, string>
      Pre_tokenizer: JObject
      Post_processor: string
      Decoder: JObject
      Model: ModelTokenizer }

//load the json with Newtonsoft.Json

let tokenizerEncoder =
    Newtonsoft.Json.JsonConvert.DeserializeObject<Tokenizer>(jsontxt)


// self.decoder = {v: k for k, v in self.encoder.items()}
// self.errors = errors  # how to handle errors in decoding
// self.byte_encoder = bytes_to_unicode()
// self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
// with open(merges_file, encoding="utf-8") as merges_handle:
//     bpe_merges = merges_handle.read().split("\n")[1:-1]
// bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
// self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
// self.cache = {}
// self.add_prefix_space = add_prefix_space

// # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
// self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

let decoder =
    tokenizerEncoder.Model.Vocab
    |> Seq.map (fun (KeyValue (k, v)) -> v, k)
    |> Map.ofSeq

let bpe_merges =
    [ for merge in tokenizerEncoder.Model.Merges do
          let m = merge.Split(' ')
          (m.[0], m.[1]) ]
//let merges_file = IO.File.ReadAllLines @"D:\\Downloads\\NeuralNets\\opt-1.3B\\merges.txt" |> Array.tail
//in python what does range(5) evaluate to?
//Answer: [0; 1; 2; 3; 4]
let bpe_ranks =
    bpe_merges
    |> Seq.mapi (fun i (a, b) -> (a, b), i)
    |> dict
    |> Dictionary

let cache = Dictionary<string, string []>()

let pat =
    new System.Text.RegularExpressions.Regex(
        """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

let vjson = tokenizerEncoder.Model.Vocab

@"D:\Downloads\NeuralNets\opt-1.3B\vocab.json"
|> IO.File.ReadAllText
|> Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>
(*def bpe(token):
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return token
    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float("inf")))
        print (f"bigram {bigram}")
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                print(f"try: j = {j}")
            except ValueError:
                print(f"Failed {i} {word[i:]}")
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j
                print(f"succeeded: {i} {new_word}")

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                print(f"if | {i} {j} {new_word}")
                i += 2
                printf(f"{i-2} => {i}")
            else:
                new_word.append(word[i])
                i += 1
                print(f"else {i-1} => {i} {new_word} {word[i]}")
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = " ".join(word)
    cache[token] = word
    return word*)

//in the python above what is if not pairs doing?
//Answer: if the list is empty then return the token
//So it checks if the list is empty?
//Answer: yes
//What about tuple(token), what doest that mean?
//Answer: it converts the string to a tuple
let stringf s = sprintf "%A" (Seq.toArray s)

let bpe (cache: Dictionary<_, _ []>) (bpe_ranks: Dictionary<_, _>) (token: string) =
    if cache.ContainsKey token then
        cache.[token]
    else
        let word = [| for c in token -> string c |]
        let pairs = get_pairs word

        if pairs.Count = 0 then
            //printfn "no pairs"
            [| token |]
        else
            let rec loop pairs (word: _ []) =
                //bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                let (w1, w2) =
                    pairs
                    |> Seq.minBy (fun (str1, str2) ->
                        bpe_ranks.tryFindIt (str1, str2)
                        |> Option.defaultValue Int32.MaxValue)
                // printfn $"bigram: {w1} {w2}"
                if not (bpe_ranks.ContainsKey((w1, w2))) then
                    // printfn "not in bpe_ranks"
                    word
                else
                    let mutable new_word = ResizeArray()
                    let mutable exitLoop = false
                    let mutable i = 0

                    while i < word.Length && not exitLoop do
                        // Console.ReadLine()
                        // printfn "word.[i..] |> Seq.tryFindIndex (fun str -> str = w1) = %A | word.[i..] = %A | w1 = %A | i = %A" (word.[i..] |> Seq.tryFindIndex (fun str -> str = w1)) (word.[i..]) w1 i
                        //j = word.index(first, i)
                        //do succeed case first
                        match Array.IndexOf(word, w1, i) with
                        //found
                        | j when j >= 0 ->
                            //let j = _j + i
                            // printfn $"Index j = {j} found | {stringf word} {i} {w1}"
                            new_word.AddRange(word.[i .. j - 1])
                            i <- j
                            // printfn $"Succeeded i: {i} new_word: {stringf new_word}"
                            if i < word.Length - 1
                               && word.[i] = w1
                               && word.[i + 1] = w2 then
                                new_word.Add(w1 + w2)
                                // printfn $"if | i: {i} j: {j} new_word: {stringf new_word}"
                                i <- i + 2
                            // printfn $"{i - 2} => {i}"
                            else
                                new_word.Add(word.[i])
                                i <- i + 1
                        // printfn $"else {i - 1} => {i} new_word: {stringf new_word} word.[i]: {word.[i-1]}"
                        | _ ->
                            new_word.AddRange(word.[i..])
                            // printfn $"Failed {i} {word.[i..]}"
                            exitLoop <- true

                    let word = new_word.ToArray()
                    //print(f"new_word: {new_word} word {word}")
                    // printfn $"new_word: {new_word} word {word}"
                    if word.Length = 1 then
                        // printfn "word.Length = 1"
                        word
                    else
                        let pairs = get_pairs word
                        loop pairs word

            let word' = loop pairs word
            cache.Add(token, word')
            word'
//what is the above doing?
//Answer: it is doing the bpe algorithm to the token and returning the token
//what is the bpe algorithm?
//Answer: it is a way to break up words into subwords and then it can be used to create a vocabulary of subwords
bpe cache bpe_ranks "" // =
bpe cache bpe_ranks "hello" // = "ap ples"

//now that I have the bpe, how do I use it to get tokens?
//Answer: you need to use the bpe to get the tokens and then you need to use the tokens to get the ids
//how do i get the ids?
//Answer: you need to use the tokens to get the ids
//how
//Answer: The ids are the index of the token in the vocabulary

(*def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
        In python what do I need to import to use re?
        Answer: import re
        *)

let internal bytes =
    List.map byte ([ '!' .. '~' ] @ [ '¡' .. '¬' ] @ [ '®' .. 'ÿ' ])
    |> HashSet

let mutable internal n = 0

let internal chars =
    [| yield! Seq.map char bytes
       for b in 0uy .. 255uy do
           if not (bytes.Contains b) then
               bytes.Add b |> ignore
               yield char (256 + n)
               n <- n + 1 |]

let byte_encoder = Dictionary(dict (Seq.zip chars bytes))

let text = "apples" // = [|"ap"; "ples"|]

[| for token in "We'll" |> bpe cache bpe_ranks -> vjson[token] |]

[ for t in pat.Matches("wellĠhelloĠthere") -> t.Value.Trim() ]

[ for t in pat.Matches("wellĠhelloĠthere") do
      let tokens = t.Value.Trim() |> bpe cache bpe_ranks
      for token in tokens -> vjson[token] ]

vjson["app"]

let byte_decoder =
    byte_encoder
    |> Seq.map (fun (KeyValue (k, v)) -> (v, k))
    |> dict
    |> Dictionary

[| for b in "well hello there" |> String.toUTF8Bytes -> string byte_decoder.[b] |]
|> String.concat ""

let strToBPEchars (s: string) =
    String.toUTF8Bytes s
    |> Array.map (fun b -> string byte_decoder[b])
    |> String.concat ""

strToBPEchars "This is a test"

let bpeEncoder (s: string) =
    let tokens = [| for r in pat.Matches s -> strToBPEchars r.Value |]

    [| for t in tokens do
           let bpe_tokens = bpe cache bpe_ranks t
           for bpe_token in bpe_tokens -> vjson[bpe_token] |]


let c = bpeEncoder "This is a test"

//to decode we need to go from ids to tokens to bytes to string
//how do we go from ids to tokens?

//from ids to tokens
let idToTokenDict =
    vjson
    |> Seq.map (fun (KeyValue (k, v)) -> (v, k))
    |> dict
    |> Dictionary

let c' =
    c
    |> Array.map (fun id -> idToTokenDict[id])
    |> String.concat ""

//from tokens to bytes to string


[| for b in c' -> byte_encoder.[b] |]
|> String.decodeFromUtf8Bytes

//alright use the above to write a function that takes ids and returns a string
let bpeDecoder (ids: int []) =
    let tokens =
        ids
        |> Array.map (fun id -> idToTokenDict[id])
        |> String.concat ""

    [| for b in tokens -> byte_encoder.[b] |]
    |> String.decodeFromUtf8Bytes

bpeEncoder "well hello there, we'll be going now"
|> bpeDecoder

strToBPEchars "well hello there, we'll be going now"
|> bpe cache bpe_ranks
|> fun tokens -> [ for token in tokens -> vjson[token] ]
