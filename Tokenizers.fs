module HuggingFaceNet.Tokenizers

open Microsoft.ML.OnnxRuntime.Tensors
open System
open BlingFire
open TensorExtensions 
open System.Collections.Generic

type BatchedTokens<'n> = {
    TokenIds : 'n DenseTensor
    AttentionMasks : 'n DenseTensor
} 

type ITokenizer<'n> =
    abstract member Tokenize : string -> 'n Tensor   
    abstract member Detokenize : 'n Tensor -> string     
    abstract member BatchTokenize: string[] -> 'n BatchedTokens
    abstract member BatchDetokenize : 'n Tensor -> string []
    
     
let initBlingFire(libloc) = Runtime.InteropServices.NativeLibrary.Load(libloc)
     
let splitSentence (s:string) = 
    let sents = BlingFire.BlingFireUtils.GetSentences s |> Seq.toArray
    if int (sents[^0][^0]) = 0 then  
        sents[^0] <- sents[^0][..^1]
        sents 
    else sents
    
let splitForBatch (s:string) =
    splitSentence s 
    |> Array.map (fun a -> [|a|])  

let splitToDocumentBatch (s:string[]) =
    s |> Array.map splitSentence
    
let liftToBatch (s:string[]) =
    Array.map (fun a -> [|a|]) s  
    
let generalTokenizer (tokenizerHandle: uint64) (unkId:int) (s: string) =
    let inBytes = Text.Encoding.UTF8.GetBytes(s)
     
    let ids = Array.create inBytes.Length 0

    let outlen = BlingFireUtils2.TextToIds(tokenizerHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(), ids.Length, unkId)
     
    ids.[..outlen - 1]  


let generalDetokenizer (tokenizerHandle: uint64) (skipSpecialTokens:bool) (ids: int []) =
    let outBytes = Array.create (ids.Length * 4) 0uy
     
    let outlen = BlingFireUtils2.IdsToText(tokenizerHandle, ids.AsSpan(), ids.Length, outBytes.AsSpan(), outBytes.Length, skipSpecialTokens)
     
    //decode to string 
    Text.Encoding.UTF8.GetString(outBytes[..outlen - 1])
     
    
//batch encode strings, add attention masks to all strings
let inline attentionMaskAndPad (tokens : _ []) =
    let lens = Array.map Array.length tokens 
    let maxlen = Array.max lens 
    [|
        for i in 0..tokens.Length - 1 do
            [|
                for j in 0..maxlen - 1 do
                    if j < lens[i] then yield (tokens[i][j], 1)
                    else yield (0,0)
            |] 
    |] |> Array.map Array.unzip |> Array.unzip
    
let tokenize (startToken, stopToken) tokenizer maxlen i0 (sents: string []) =
    //Actually, I will use a buffer approach.
    //Recursive for early exit.
    let buffer = Array.create maxlen stopToken
    let tokenlenInit = 
        match startToken with 
        | None -> 0 
        | Some startToken ->  
            buffer.[0] <- startToken; 1

    let rec loop i tokenlen =
        let sent = sents.[i]
        let tokenized : int [] = tokenizer sent
        let tokenlen' = tokenlen + tokenized.Length

        if tokenlen' + 1 < maxlen then // account for stop tag
            Array.Copy(tokenized, 0, buffer, tokenlen, tokenized.Length)
            if i + 1 < sents.Length then loop (i + 1) tokenlen'
            else i + 1, tokenlen' //done
        else i, tokenlen

    let i, toklen = loop i0 tokenlenInit
    buffer.[..toklen], i
     
///Transformers have windows of size = N tokens max. To get around this, rather 
///than just truncate we split by sentence and join sentences until they exhaust the window
///and then start a new window until all windows have been complete. Think of trying to 
///write a tweet thread were all tweets must contain only complete sentences.
let tokenizeSplit tokenize maxlen (sents : _ []) =
    let rec loop i =
        [| if i < sents.Length then 
            let (s : int []), j = tokenize maxlen i sents  
            if s.Length > 2 || (s.Length = 2 && s.[1] = 1) then yield s //T5 can have len = 2, hard code this
            // if i = j and there're more sents left then a sentence was probably too long. We can either fail or skip. Given token window lens are > 300 to 600 words, I will choose to skip.
            let j' =
                if i = j then j + 1
                else j
            yield! loop j' |]
    loop 0
 

module BPE =
    let internal bytes =
        List.map byte ([ '!' .. '~' ] @ [ '¡' .. '¬' ] @ [ '®' .. 'ÿ' ])
        |> HashSet

    let mutable n = 0

    let internal chars =
        [| yield! Seq.map char bytes
           for b in 0uy .. 255uy do
               if not (bytes.Contains b) then
                   bytes.Add b |> ignore
                   yield char (256 + n)
                   n <- n + 1 |]

    let byte_encoder = Dictionary (dict (Seq.zip chars bytes))

type BlingFireTokenizer(loc, startToken, stopToken, maxlen, unknownId, ?detokenizer) =

    let tokHandle = BlingFireUtils2.LoadModel loc

    let appendStartEndTokens s =
        [| match startToken with
           | None -> ()
           | Some startToken -> yield startToken
           yield! s
           yield stopToken |] 

    member __.TokenizeToArray(s: string) = generalTokenizer tokHandle unknownId s

    member __.Detokenize(ids: int [], ?skipSpecialTokens) =
        let skiptok = (defaultArg skipSpecialTokens true)  
        match detokenizer with 
        | None -> failwith "No detokenizer provided"
        | Some detokenizer -> detokenizer skiptok ids 
        
    member __.TokenizeSplit(s: string []) =
        let tokens =
            tokenizeSplit (tokenize (startToken, stopToken) (generalTokenizer tokHandle unknownId)) maxlen s

        attentionMaskAndPad tokens

    member __.Tokenize(s: string) =
        generalTokenizer tokHandle unknownId s
        |> appendStartEndTokens
        |> Array.toTensor

    member t.BatchTokenize(s: string []) =
        let tks, masks = t.TokenizeSplit s

        { TokenIds = Array2D.toTensor (array2D tks)
          AttentionMasks = Array2D.toTensor (array2D masks) }

    member t.BatchTokenize(s: string) =
        let strs = BlingFireUtils.GetSentences s |> Seq.toArray
        let tks, masks = t.TokenizeSplit strs

        { TokenIds = Array2D.toTensor (array2D tks)
          AttentionMasks = Array2D.toTensor (array2D masks) }

    member t.BatchTokenize(s: string [] []) =
        let mutable k = -1

        let flattened =
            [| for i in 0 .. s.Length - 1 do 
                let tokens =
                    tokenizeSplit
                        (tokenize 
                            (startToken, stopToken) 
                            (generalTokenizer tokHandle unknownId)) maxlen s[i]

                for j in 0 .. tokens.Length - 1 do
                    k <- k + 1
                    (k, i), tokens[j] |]

        let lookup, flatids = Array.unzip flattened
        let paddedIds, masks = attentionMaskAndPad flatids

        dict lookup,
        { TokenIds = Array2D.toTensor (array2D paddedIds)
          AttentionMasks = Array2D.toTensor (array2D masks) }

    member t.BatchDetokenize(ids: int [] []) = Array.map t.Detokenize ids

    member t.BatchDetokenize(ids: Tensor<int>) =
        Tensor.toJaggedArray2D ids |> t.BatchDetokenize

    member __.Dispose() =
        BlingFireUtils.FreeModel(tokHandle) |> ignore

    static member NewBertTokenizer(loc) =
        new BlingFireTokenizer(IO.Path.Combine(loc, "bert_base_tok.bin"), Some 101, 102, 512, 100)

    interface ITokenizer<int> with
        member t.Tokenize(s: string) = t.Tokenize s
        member t.BatchTokenize(s: string []) = t.BatchTokenize s
        member t.Detokenize(ids: Tensor<int>) = t.Detokenize(Tensor.toArray ids)
        member t.BatchDetokenize(ids: Tensor<int>) = t.BatchDetokenize ids

    interface IDisposable with
        member t.Dispose() = t.Dispose()