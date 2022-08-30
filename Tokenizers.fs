module TensorNet.Tokenizers

open Microsoft.ML.OnnxRuntime.Tensors
open System
open BlingFire
open TensorExtensions 
open System.Collections.Generic

[<RequireQualifiedAccess>]
type PrependType =
    | Multiple of string []
    | Single of string 
    
type Tokenized<'n> = {
    TokenIds : 'n DenseTensor
    AttentionMasks : 'n DenseTensor
} 

type BatchedTokens<'n> = {
    Indices : int []
    TokensAndMasks : Tokenized<'n>
} 

type ITokenizer<'n> =
    abstract member RawTokenize : string -> int []
    abstract member Detokenize : 'n Tensor -> string     
    abstract member Tokenize: string[] -> 'n Tokenized     
    abstract member BatchTokenize: string[][] * int * string[] -> 'n BatchedTokens    
    abstract member BatchDetokenize : 'n Tensor -> string []
    

module BlingFireUtils = 
    let splitSentence (str:string) =  
        let sents = BlingFireUtils.GetSentences str |> Seq.toArray 
        //I'm not sure why blingfire adds a weird token to the last sentence
        sents.[^0] <- sents.[^0].[..^1]
        sents 
        
        
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
let inline attentionMaskAndPad (tokens: _ []) =
    let lens = Array.map Array.length tokens
    let maxlen = Array.max lens

    [| for i in 0 .. tokens.Length - 1 do
        [| for j in 0 .. maxlen - 1 do
            if j < lens[i] then
                yield (tokens[i][j], 1)
            else
                yield (0, 0) |] |]
    |> Array.map Array.unzip
    |> Array.unzip    
    
let tokenize (startToken, stopToken, septoken) prepended tokenizer maxlen i0 (sents: string []) =
    //Actually, I will use a buffer approach.
    //Recursive for early exit.
    let buffer = Array.create maxlen stopToken
    let tokenlenInit = 
        let tokenlen =
            match startToken with 
            | None -> 0 
            | Some startToken ->  
                buffer.[0] <- startToken; 1
        match prepended with 
        | None -> tokenlen 
        | Some txt -> 
            let prependedTokens : int [] = tokenizer txt
            Array.Copy(prependedTokens, 0, buffer, tokenlen, prependedTokens.Length)
            match septoken with 
            | None -> tokenlen + prependedTokens.Length + 1
            | Some sep -> 
                buffer[tokenlen + prependedTokens.Length + 1] <- sep
                tokenlen + prependedTokens.Length + 2 //1 + 1

    let rec loop i tokenlen =
        let sent = sents.[i]
        let tokenized = tokenizer sent
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
///and then start a new window until all text has been consumed. Think of trying to 
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
 
let appendStartEndTokens (startToken, stopToken) s =
    [| match startToken with
       | None -> ()
       | Some startToken -> yield startToken
       yield! s
       yield stopToken |] 


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
     
     
type GeneralTokenizer(startToken, stopToken, maxlen, ?sepToken, ?endOfSentenceTokens) = 
    let specialTokens =
        HashSet [ 
            match startToken with
            | None -> ()
            | Some t -> yield t
            match stopToken with
            | None -> ()
            | Some t -> yield t 
            match sepToken with
            | None -> ()
            | Some t -> yield t 
        ] 
   
    let stoptoken = match stopToken with Some stop -> stop | None -> startToken.Value
    
    ///This method is for handling the case where you have a series of documents which exceed the window size and
    ///tracking where each split document cames from
    let flattenTokenBatch maxlen start prepends tokenizer (s: _ []) =
        let flattened =
            [| for i in 0 .. s.Length - 1 do
                   let tokens =
                       tokenizeSplit
                           (tokenize
                               (startToken, stoptoken, sepToken)
                               (Option.map
                                   (function
                                   | PrependType.Multiple ps -> ps[i]
                                   | PrependType.Single p -> p)
                                   prepends)
                               tokenizer)
                           maxlen
                           s[i]

                   for j in 0 .. tokens.Length - 1 do
                       start + i, tokens[j] |]

        let lookup, flatids = Array.unzip flattened
        let paddedIds, masks = attentionMaskAndPad flatids

        { Indices = lookup
          TokensAndMasks =
            { TokenIds = Array2D.toTensor (array2D paddedIds)
              AttentionMasks = Array2D.toTensor (array2D masks) } }   
              
              
    member __.SpecialTokens = specialTokens

    member __.EndOfSentenceTokens = endOfSentenceTokens

    abstract member Tokenizer : string -> int[]
    default __.Tokenizer s = failwith "not implemented"
    
    abstract member Detokenize : int[] * ?skipSpecialTokens:bool -> string
    default __.Detokenize(ids: int [], ?skipSpecialTokens:bool) = 
        failwith "not implemented"

    member private t.TokenizeSplitAndPad(s: string [], ?prepend) =
        let tokens =
            tokenizeSplit
                (tokenize (startToken, stoptoken, sepToken) prepend t.Tokenizer)
                maxlen
                s

        attentionMaskAndPad tokens

    member t.RawTokenize(s: string) =
        match stopToken with 
        | None -> 
            Array.append [|if startToken.IsSome then yield startToken.Value|] (t.Tokenizer s) 
        | Some stoptoken ->
            t.Tokenizer s
            |> appendStartEndTokens (startToken, stoptoken)

    member t.RawTokenizeTruncate(s: string) =
        let res =
            match stopToken with 
            | None -> 
                Array.append [|if startToken.IsSome then yield startToken.Value|] (t.Tokenizer s) 
            | Some stoptoken ->
                t.Tokenizer s
                |> appendStartEndTokens (startToken, stoptoken)
        if res.Length > maxlen then
            Result.Ok(res.[..maxlen - 1])
        else
            Result.Error res        
        
    member t.Tokenize(s: string seq, ?prepend) =
        let tks, masks = t.TokenizeSplitAndPad (Seq.toArray s, ?prepend = prepend)

        { TokenIds = Array2D.toTensor (array2D tks)
          AttentionMasks = Array2D.toTensor (array2D masks) }

    member t.Tokenize(s: string, ?prepend) =
        let strs = BlingFireUtils.splitSentence s |> Seq.toArray
        let tks, masks = t.TokenizeSplitAndPad (strs, ?prepend = prepend)

        { TokenIds = Array2D.toTensor (array2D tks)
          AttentionMasks = Array2D.toTensor (array2D masks) }

    member t.BatchTokenize(s: string [][], ?startindex, ?prepends) =
        let start = defaultArg startindex 0

        flattenTokenBatch
            maxlen
            start 
            prepends
            t.Tokenizer
            s

    member t.BatchTokenize(strs: string seq, ?startindex: int) =
        let strs = strs |> Seq.toArray |> Array.map (BlingFireUtils.splitSentence >> Seq.toArray)
        
        t.BatchTokenize(strs, ?startindex = startindex)
        
    member t.BatchTokenize(prepends: string seq, strs: string seq, ?startindex: int) =
        let strs = strs |> Seq.toArray |> Array.map (BlingFireUtils.splitSentence >> Seq.toArray)
        
        t.BatchTokenize(strs, ?startindex = startindex, ?prepends = Some(PrependType.Multiple (Seq.toArray prepends)))
        
    member t.BatchTokenize(prepend: string, strs: string seq, ?startindex: int) =
        let strs = strs |> Seq.toArray |> Array.map (BlingFireUtils.splitSentence >> Seq.toArray)
        t.BatchTokenize(strs, ?startindex = startindex, ?prepends = Some(PrependType.Single prepend))
        
    member t.BatchTokenize(s: string [] [], prepends: string [], ?startindex: int) =
        t.BatchTokenize(s, ?startindex = startindex, ?prepends = Some(PrependType.Multiple prepends))

    member t.BatchTokenize(s: string [] [], prepend: string, ?startindex: int) =
        t.BatchTokenize(s, ?startindex = startindex, ?prepends = Some(PrependType.Single prepend))

    member t.BatchDetokenize(ids: int [] []) = Array.map t.Detokenize ids

    member t.BatchDetokenize(ids: Tensor<int>) =
        Tensor.toJaggedArray2D ids |> t.BatchDetokenize

    interface ITokenizer<int> with
        member t.RawTokenize(s: string) = t.RawTokenize s
        member t.Tokenize(s: string []) = t.Tokenize s 
        member t.BatchTokenize(s: string [][], startindex: int, prepends: string []) =
            t.BatchTokenize(s, prepends, startindex) 

        member t.Detokenize(ids: Tensor<int>) = t.Detokenize(Tensor.toArray ids)
        member t.BatchDetokenize(ids: Tensor<int>) = t.BatchDetokenize ids          

         
type BlingFireTokenizer(loc, startToken, stopToken, maxlen, unknownId, ?detokenizer, ?sepToken, ?endOfSentenceTokens) =
    inherit GeneralTokenizer(startToken, stopToken, maxlen, ?sepToken = sepToken, ?endOfSentenceTokens = endOfSentenceTokens)
    let tokHandle = BlingFireUtils2.LoadModel loc

    member __.TokenizerHandle = tokHandle

    override t.Tokenizer s =
        generalTokenizer tokHandle unknownId s

    override t.Detokenize(ids: int [], ?skipSpecialTokens) =
        let skiptok = (defaultArg skipSpecialTokens true)

        match detokenizer with
        | None -> failwith "No detokenizer provided"
        | Some detokenizer -> detokenizer skiptok t.SpecialTokens ids
         
    member __.Dispose() =
        BlingFireUtils.FreeModel(tokHandle) |> ignore
    
    static member NewBertTokenizer(loc) =
        new BlingFireTokenizer(IO.Path.Combine(loc, "bert_base_tok.bin"), Some 101, Some 102, 512, 100)

    static member NewRobertaTokenizer(loc) =
        new BlingFireTokenizer(IO.Path.Combine(loc, "roberta.bin"), Some 0, Some 2, 512, 3, sepToken = 2) 

    static member NewGPT2Tokenizer(loc,?windowsize) =
        new BlingFireTokenizer(IO.Path.Combine(loc, "gpt2.bin"), Some 2, None, defaultArg windowsize 512, 2, endOfSentenceTokens = [|764; 5145; 5633; 220|])
    
    interface IDisposable with
        member t.Dispose() = t.Dispose()
         

type SentencePieceTokenizer(loc, startToken, stopToken, maxlen, ?sepToken, ?endOfSentenceTokens) =
    inherit GeneralTokenizer(startToken, stopToken, maxlen, ?sepToken = sepToken, ?endOfSentenceTokens = endOfSentenceTokens)
    let spiece = new SentencePieceDotNET.SentencePieceDotNET()

    do spiece.Load(loc)

    override t.Tokenizer s = spiece.Encode(s) 

    override t.Detokenize(ids: int [], ?skipSpecialTokens) =
        match skipSpecialTokens with
        | Some true ->
            ids
            |> Array.filter (t.SpecialTokens.Contains >> not)
        | _ -> ids
        |> spiece.Decode
                    
    member t.Dispose() = spiece.Dispose() 
    
    interface IDisposable with
        member t.Dispose() = t.Dispose()
     
    static member NewT5Tokenizer(loc) =
        new SentencePieceTokenizer(loc, None, Some 1, 512, endOfSentenceTokens = [|5; 55; 58|])
    
    static member NewDebertaTokenizer(loc) =
        new SentencePieceTokenizer(loc, Some 1, Some 2, 512)
          
        

    

    
    
            
    