#r "System.Runtime.InteropServices" 
#I @"C:\Users\cybernetic\source\repos\"
#r @"BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#r @"SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net472\SentencePieceDotNET.dll" 
#load "tensor-extensions.fsx"
#load "TestDocSnippets.fsx"
#time "on"

open Newtonsoft.Json
open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 
open BlingFire
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open ``Tensor-extensions``

let vocab =
    let vocabDict =
        JsonConvert.DeserializeObject<Collections.Generic.Dictionary<string, int>>
            (IO.File.ReadAllText
                 @"D:\Downloads\NeuralNets\roberta-base-squad2\vocab.json")
    [| for KeyValue(w, _) in vocabDict -> w |]

//Test that the library is working at all.
Runtime.InteropServices.NativeLibrary.Load(@"D:Downloads\blingfiretokdll.dll")

let robertaBin = BlingFireUtils.LoadModel  @"C:/Users/cybernetic/source/repos/BlingFire/bin/Release/net5.0/Files/roberta.bin"

///////////
// allocate space for ids 
let ids = Array.create 1024 0

let inBytes = Text.Encoding.UTF8.GetBytes("hello this is a test</s>");

let _ = BlingFireUtils.TextToIds (robertaBin, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(),  ids.Length, 0)

ids

//Test sentence splitting
BlingFireUtils.GetSentences "This is a test of 3.45 people with Ph.D's said\n Mr. Robert. Who will take this."
|> Seq.toArray
|> printfn "%A"

/////////////////////////////

let startToken = [|0|]

let stopToken = [|2|]

let appendStartEndTokens s = Array.concat [startToken; s; stopToken]

///The current blingfire tokenizer behaves as hugging-face's with ``add_prefix_space=True``
let genericEncoder (tokenizerHandle: uint64) (s: string) =
    let inBytes = Text.Encoding.UTF8.GetBytes(s)

    let ids = Array.create inBytes.Length 0

    let outlen = BlingFireUtils.TextToIds(tokenizerHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(), ids.Length, 3)
     
    ids.[..outlen - 1] 
    
///The Roberta Transformer has a window of 512 tokens max. To get around this, rather 
///than just truncate we split every 510 tokens and let the user decide downstream what to do from there
let robertaEncoder (tokenizerHandle: uint64) (s: string) =
    genericEncoder tokenizerHandle s 
    |> Array.split_TakeN_atATime 510

///The BART Transformer has a window of 1024 tokens max. To get around this, rather 
///than just truncate we split every 1022 tokens and let the user decide downstream what to do  
let bartEncoder (tokenizerHandle: uint64) (s: string) =
    genericEncoder tokenizerHandle s 
    |> Array.split_TakeN_atATime 1022

let robertaDecoder (vocab:string []) tokens = 
    let inline lookUpFixHack (s:string) = s.Replace("âĢĻ", "’").Trim(' ')
    tokens 
    |> Seq.fold (fun str t -> 
        let w = vocab.[t] 
        if w.[0] = 'Ġ' then str + " " + w.[1..] 
        else str + w) ""
    |> lookUpFixHack 

//////////////////

let toks = robertaEncoder robertaBin "I don't think that's ribonucleic"
        
robertaDecoder vocab toks.[0]

///Appending tokens test
let w1 = robertaEncoder robertaBin "Hello there" |> Array.head

Array.concat [startToken; w1; stopToken] |> robertaDecoder vocab

////////////////////

// The problem of decoding unicode characters. Using vocab.json seems to result in mangling and using a lookup table will not scale.
//How to fix? State: Don't know.
//

[|for t in robertaEncoder robertaBin "漢" |> Array.head do 
    yield! vocab.[t].Replace("Ġ", "").ToCharArray() |]
|> Array.map byte 
|> Strings.DecodeFromUtf8Bytes

Strings.toUTF8Bytes "漢"

[|for t in robertaEncoder robertaBin "’" |> Array.head do 
    yield! vocab.[t].Replace("Ġ", "").ToCharArray() |]
|> Array.map byte 
|> Strings.DecodeFromUtf8Bytes

[|for t in robertaEncoder robertaBin "Inside the apple’s" |> Array.head do 
    yield vocab.[t].Replace("Ġ", "")|]

Strings.toUTF8Bytes "’"

Strings.toUTF8Bytes "ë"

Array.map2 (-) [|229uy; 173uy; 151uy|] [|229uy; 67uy; 57uy|]

Array.map2 (-) [|226uy; 128uy; 153uy|] [|226uy; 34uy; 59uy|]

Array.map2 (-) [|226uy; 134uy; 144uy|] [|226uy; 40uy; 50uy|]

[|for t in robertaEncoder robertaBin "字" |> Array.head -> vocab.[t]|]

[|for t in robertaEncoder robertaBin "漢" |> Array.head -> vocab.[t]|]

"æ¼¢åŃĹ".ToCharArray() |> Array.map byte |> Strings.DecodeFromUtf8Bytes

///////////////////////// 
 
let robertaQA = new InferenceSession(@"D:\Downloads\NeuralNets\roberta-base-squad2\roberta-base-squad2.onnx")
 
robertaQA.OutputMetadata

//////////////////////////////////////


let formatforSquadQATruncate (context:string, question:string) = 
    let truncate (s:string) =   
        let split = s.Split(' ')
        if split.Length > 360 then split.[..359] |> String.concat " " else s

    $"<s> {question} </s> <s> {truncate context} </s>"

let formatforSquadQA (context:string, question:string) = 
    let splitContext = context.Split(' ') |> Array.split_TakeN_atATime 360  
    [|for context in splitContext -> 
        let bcontext = String.concat " " context
        $"<s> {question} </s> <s> {bcontext} </s>"|] 

let runrobertaQA (tokens:int []) =
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|tokens|])

    let outputs = robertaQA.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    let res =
        [| for output in outputs -> 
            let result = output.AsTensor<float32>()

            let dims = result.Dimensions.ToArray()

            let vec = [| for i in 0..dims.[1] - 1 -> result.[0, i] |]
            
            output.Dispose()
            vec |]
    
    outputs.Dispose()

    {|Start = res.[0]; End = res.[1]|}

////////////////////

let c = "I picked an apple, ate it then went next door to pick a banana."

let qas = 
    formatforSquadQA (c, 
        "What is the main idea of this research ?")
    |> Array.map (robertaEncoder robertaBin >> Array.head)
let answers = Array.map runrobertaQA qas

[| for n in 0 .. 2 do
    [| for (i, a) in Array.zip [| 0 .. answers.Length - 1 |] answers do
        let start, sp = Array.nthTop n a.Start
        let endIndex, ep = Array.nthTop n a.End

        if exp sp > 1.f then
            let str = qas.[i].[start..endIndex] |> robertaDecoder vocab

            if str <> "" && not (str.Contains "<s>") then
                yield str, exp sp, exp ep |] |]


BlingFireUtils.GetWords c 
|> Seq.toArray 
|> Array.countBy id 
|> Array.sortByDescending snd 


let qas =  
    formatforSquadQA ("I picked an apple, ate it then went next door to pick a banana.", 
        "What door did I go through ?")
    |> Array.map (robertaEncoder robertaBin >> Array.head)
let answers = Array.map runrobertaQA qas

answers.[0].Start 
|> Array.map exp 
|> Array.normalize
|> Array.mapi (fun i p -> robertaDecoder vocab [qas.[0].[i]], p)
|> Prelude.TextHistogram.genericHistogram float 20 


/////////////////////////
//Stats on token size vs words len

robertaEncoder robertaBin c |> Array.concat |> Array.length

c.Split(' ') |> Seq.length

[581./727.; 788./1032.;244./322.; 530./741.; 565./771.; 2073./ 3270. ]

"âĸ²".ToCharArray() |> Array.map byte //|> Strings.DecodeFromUtf8Bytes


let deMangleUnicode (str:string) =
    str.ToCharArray()
    |> Array.mapi (fun i c ->
        let b = byte c
        if i = 0 then b
        else 
            if (int b) + 94 > 255 then b
            else b + 94uy)
    |> Strings.DecodeFromUtf8Bytes

["âĢĿ"; "âĸ²"; "åŃĹ"; "âĢľ";"æ¼¢"]
|> List.map deMangleUnicode

Strings.toUTF8Bytes "▲"

Seq.map Char.IsLetterOrDigit "æ¼¢"

["âĢĿ"; "âĸ²"; "åŃĹ"; "âĢľ"] |> List.map (Seq.map Char.IsLetterOrDigit)

/////////////////////// 
//distilbart-cnn-6-6 bart-large-cnn
let bartModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-encoder.onnx")

let bartModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-decoder.onnx")

let tok0 = bartEncoder robertaBin "This is a test." 

let tok0 = [|[|152; 16; 10; 1296; 1437|]|]
 
bartEncoder robertaBin ("A\nB")

let c = IO.File.ReadAllText (pathCombine DocumentsFolder "doc.txt")

let toks = bartEncoder robertaBin c |> Array.map appendStartEndTokens

let t = Tensors.ArrayTensorExtensions.ToTensor(array2D tok0)

t

robertaDecoder vocab tok0.[0]

let outputs = bartModelEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]

let encoderStates = Seq.head outputs

encoderStates.Name <- "encoder_hidden_states"
 
robertaDecoder vocab [50118] 

outputs.Dispose()

encoderStates.Dispose() 

//distilbartxsumDecoder.InputMetadata
//generated |> Tensor.toArray2D |> Array2D.map float32 |> Single.DenseMatrix.OfArray
//result |> Tensor.toMatrix
//result |> Tensor.toJaggedArray2D

let greedySample (decoder:InferenceSession) maxlen encoderStates =
    
    let generatedTokens = ResizeArray startToken
    let mutable generated = Tensors.ArrayTensorExtensions.ToTensor(array2D [| startToken |])

    let mutable c = 0
    let mutable stop = false
    let sw = Diagnostics.Stopwatch()

    while c < maxlen && not stop do 
        let logits =
            decoder.Run 
                [| NamedOnnxValue.CreateFromTensor("input_ids", generated)
                   encoderStates |] 

        let decoded = Seq.head logits

        let result = decoded.AsTensor<float32>()   
         
        let index = Array.argmaxf32 result.[0, ^0, *] 

        if index <> stopToken.[0] then generatedTokens.Add index  
        
        generated <- 
            Tensors.ArrayTensorExtensions.ToTensor(array2D [| generatedTokens.ToArray() |])
             
        decoded.Dispose()

        logits.Dispose()

        sw.Stop() 
        c <- c + 1
        stop <- index = stopToken.[0]

    robertaDecoder vocab (generatedTokens.ToArray())

greedySample bartModelDecoder 50 encoderStates

//TODO improve splitting

let runSummary (encoder:InferenceSession) (decoder:InferenceSession) seqlen toks =
    
    [|
        for tok in toks do
            let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|tok|])  
    
            let outputs = encoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    
            let encoderStates = Seq.head outputs
    
            encoderStates.Name <- "encoder_hidden_states"

            let str = greedySample decoder seqlen encoderStates

            encoderStates.Dispose()
            outputs.Dispose() 
            str.Replace("<s>", "")
            yield str
    |] 
let bartModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-encoder.onnx")

let bartModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-decoder.onnx")

let bartModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\distilbart-cnn-6-6\distilbart-cnn-6-6-encoder.onnx")

let bartModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\distilbart-cnn-6-6\distilbart-cnn-6-6-decoder.onnx")

let toks = bartEncoder robertaBin c |> Array.map appendStartEndTokens

let ss1 = runSummary bartModelEncoder bartModelDecoder 200 toks

ss1