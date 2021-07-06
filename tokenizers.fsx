#I @"C:\Users\cybernetic\source\repos\"
#I @"C:\Users\cybernetic\.nuget\packages\"
#r @"BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"

open Prelude.Common
open BlingFire
open System
  

module BlingFireUtils = 
    let splitSentence (str:string) =  
        let sents = BlingFireUtils.GetSentences str |> Seq.toArray 
        //I'm not sure why blingfire adds a weird token to the last sentence
        sents.[^0] <- sents.[^0].[..^1]
        sents 


//Test that the library is working at all.
let initBlingFire() =
    Runtime.InteropServices.NativeLibrary.Load(@"D:\Downloads\blingfiretokdll.dll")

///The current blingfire tokenizer behaves as hugging-face's with ``add_prefix_space=True``
let generalTokenizer (tokenizerHandle: uint64) (s: string) =
    let inBytes = Text.Encoding.UTF8.GetBytes(s)

    let ids = Array.create inBytes.Length 0

    let outlen = BlingFireUtils.TextToIds(tokenizerHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(), ids.Length, 3)
     
    ids.[..outlen - 1] 

(*
All right so the basic idea here is that we split the document up into sentences and then tokenize each sentence. 
There are several ways to go about this. One way is to recurse over the sentence and have a temporary stack of the current set of tokens and then a more permanent stack of all the full token windows. 

The problem is that each of these arrays of tokens need to be concatenated but if we're doing that we might as well do it more clearly. One way to do this is to have an inner function build a current sentence so far and then returns the sentences that no longer fit and yields that array. Then recurse again on the rest of the sentences, repeat the above and so on and so forth recursively.
*) 
type TokenizerInfo =
    { StartToken: int option
      StopToken: int } 
 
let tokenize tokenizerInfo tokenizer maxlen i0 (sents: string []) =
    //Actually, I will use a buffer approach.
    //Recursive for early exit.
    let buffer = Array.create maxlen tokenizerInfo.StopToken
    let tokenlenInit = 
        match tokenizerInfo.StartToken with 
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