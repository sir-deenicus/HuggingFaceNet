module TensorNet.BPE

open System.Collections.Generic
open Newtonsoft.Json.Linq
open System
open Prelude.Common
open Newtonsoft.Json

type BPEModelData =
    { Type: string
      Dropout: string
      Unk_token: string
      Continuing_subword_prefix: string
      End_of_word_suffix: string
      Fuse_unk: bool
      Vocab: Dictionary<string, int>
      Merges: string [] }

type BPETokenizerData =
    { Version: string
      Truncation: string
      Padding: string
      Added_tokens: JObject []
      Normalizer: Dictionary<string, string>
      Pre_tokenizer: JObject
      Post_processor: JObject
      Decoder: JObject
      Model: BPEModelData }


let get_pairs (word: string []) =
    // Return set of symbol pairs in a word.
    // Word is represented as tuple of symbols (symbols being variable-length strings).
    let pairs = HashSet()
    let mutable prev_char = string word.[0]

    for str in word.[1..] do
        pairs.Add((prev_char, str)) |> ignore
        prev_char <- str

    pairs


let build_bpe_merges (tokenizer: BPETokenizerData) =
    [ for merge in tokenizer.Model.Merges do
          let m = merge.Split(' ')
          (m.[0], m.[1]) ]

let bpe_ranks (bpe_merges: list<_>) =
    bpe_merges
    |> List.mapi (fun i (a, b) -> (a, b), i)
    |> dict
    |> Dictionary
  
let pat =
    new System.Text.RegularExpressions.Regex(
        """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

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


let internal bytes =
    List.map byte ([ '!' .. '~' ] @ [ '¡' .. '¬' ] @ [ '®' .. 'ÿ' ])
    |> HashSet

let internal chars =
    let mutable n = 0

    [| yield! Seq.map char bytes
       for b in 0uy .. 255uy do
           if not (bytes.Contains b) then
               bytes.Add b |> ignore
               yield char (256 + n)
               n <- n + 1 |]

let byte_encoder = Dictionary(dict (Seq.zip chars bytes))

let byte_decoder =
    byte_encoder
    |> Seq.map (fun (KeyValue (k, v)) -> (v, k))
    |> dict
    |> Dictionary


let strToBPEchars (s: string) =
    String.toUTF8Bytes s
    |> Array.map (fun b -> string byte_decoder[b])
    |> String.concat ""

let bpeEncoder bpe_ranks cache (vjson:Dictionary<_,int>) (s: string) =
    let tokens = [| for r in pat.Matches s -> strToBPEchars r.Value |]

    [| for t in tokens do
           let bpe_tokens = bpe cache bpe_ranks t
           for bpe_token in bpe_tokens -> vjson[bpe_token] |]
  

let bpeDecoder
    (addedTokens: Dictionary<_, _>)
    (idToTokenDict: Dictionary<_, _>)
    (byte_encoder: Dictionary<_, _>)
    (ids: int [])
    =
    let tokens =
        ids
        |> Array.map (fun id ->
            match idToTokenDict.tryFind id with
            | Some token -> token
            | None ->
                match addedTokens.tryFind id with
                | Some token -> token | None -> "?")
        |> String.concat ""

    [| for b in tokens ->
           //if it's not in byte_encoder, this is control character, so keep it as it is
           match byte_encoder.tryFind b with
           | Some b -> b
           | None -> byte b |]
    |> String.decodeFromUtf8Bytes
