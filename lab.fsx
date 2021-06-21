#r "System.Memory"
#r "System.Runtime.InteropServices"
#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\net5\Prelude.dll"
#r @"D:\Downloads\NeuralNets\onnx\Microsoft.ML.OnnxRuntime.dll"
#r @"C:\Users\cybernetic\source\repos\SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net472\SentencePieceDotNET.dll"
#r @"C:\Users\cybernetic\.nuget\packages\newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
#r @"C:\Users\cybernetic\source\repos\BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"D:\Downloads\NeuralNets\onnx\Microsoft.ML.OnnxRuntime.dll"
#r @"C:\Users\cybernetic\.nuget\packages\newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"

open Newtonsoft.Json
open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 
open BlingFire

let vocab =
    let vocabDict =
        JsonConvert.DeserializeObject<Collections.Generic.Dictionary<string, int>>
            (IO.File.ReadAllText
                 @"D:\Downloads\NeuralNets\roberta-base-squad2\vocab.json")
    [| for KeyValue(w, _) in vocabDict -> w |]

//Test that the library is working at all.

Runtime.InteropServices.NativeLibrary.Load(@"D:Downloads\blingfiretokdll.dll")

//Library seems to not like paths
let robertaBin = BlingFireUtils.LoadModel  @"C:/Users/cybernetic/source/repos/BlingFire/bin/Release/net5.0/Files/roberta.bin"

///////////
// allocate space for ids and offsets
let ids = Array.create 1024 0

let inBytes = Text.Encoding.UTF8.GetBytes("hello this is a test</s>");

let _ = BlingFireUtils.TextToIds (robertaBin, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(),  ids.Length, 0)

ids

//Test sentence splitting
BlingFireUtils.GetSentences "This is a test of 3.45 people with Ph.D's said\n Mr. Robert. Who will take this."
|> Seq.toArray
|> printfn "%A"

//The current blingfire tokenizer behaves as hugging-face's with ``add_prefix_space=True``
///The Roberta Transformer has a window of 512 tokens max. To get around this, rather 
///than just truncate we split every 500 tokens and let the user decide downstream what to do from there
///500 instead of 512 to allow some appending
let robertaEncoder (tokenizerHandle: uint64) (s: string) =
    let inBytes = Text.Encoding.UTF8.GetBytes(s)

    let ids = Array.create inBytes.Length 0

    let outlen = BlingFireUtils.TextToIds(tokenizerHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(), ids.Length, 0)
     
    ids.[..outlen - 1] 
    |> Array.split_TakeN_atATime 500


let robertaDecoder (vocab:string []) tokens = 
    let inline lookUpFixHack (s:string) = s.Replace("âĢĻ", "’").Trim(' ')
    tokens 
    |> Seq.fold (fun str t -> 
        let w = vocab.[t] 
        if w.[0] = 'Ġ' then str + " " + w.[1..] 
        else str + w) ""
    |> lookUpFixHack 

let toks = robertaEncoder robertaBin "I don't think that's ribonucleic"
        
robertaDecoder vocab toks.[0]

////////////////////

// The problem of decoding unicode characters. Using vocab.json seems to result in mangling and using a lookup table will not scale.
//How to fix? State: Don't know.

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

[|for t in robertaEncoder robertaBin "漢字" |> Array.head -> vocab.[t]|]

"æ¼¢åŃĹ".ToCharArray() |> Array.map byte |> Strings.DecodeFromUtf8Bytes

/////////////////////////

let robertaQA = new InferenceSession(@"D:\Downloads\NeuralNets\roberta-base-squad2\roberta-base-squad2.onnx")

robertaQA.OutputMetadata
 
#time 


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

let qas = 
    formatforSquadQA (c0, "What will AWS do ? ")
    |> Array.map (robertaEncoder robertaBin >> Array.head)

let w1 = robertaEncoder robertaBin "Hello there" |> Array.head

let startToken = [|0|]
let endToken = [|2|]

Array.concat [startToken; w1; endToken] |> robertaDecoder vocab
robertaEncoder robertaBin c |> Array.concat |> Array.length
c.Split(' ') |> Seq.length

360. * 1./0.76

[581./727.; 788./1032.;244./322. ]

robertaEncoder robertaBin "T "
BlingFireUtils.GetWords c |> Seq.length

let q = encodeQuestion (robertaEncoder robertaBin) "What shall I eat ?" "I picked an apple, ate it then went next door to pick a banana."

let q = encodeQuestion (robertaEncoder robertaBin) "What is this article about ?" c

let qas = 
    formatforSquadQA (c, "What is this article about ? ")
    |> Array.map (robertaEncoder robertaBin >> Array.head)


#time "on"

let argmax d = Array.indexed d |> Array.maxBy snd |> fst

let inline nthMax n d =
    Array.indexed d 
    |> Array.sortByDescending snd 
    |> Array.skip n
    |> Array.head 

"âĢĿ".ToCharArray() |> Array.map byte |> Array.mapi (fun i b -> if i > 0 then b + 94uy else b)
|> Strings.DecodeFromUtf8Bytes

let runrobertaQA (q:int []) =
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|q|])

    let outputs = robertaQA.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    let res =
        [| for output in outputs ->

            let result = output.AsTensor<float32>()

            let dims = result.Dimensions.ToArray()
    
            [| for i in 0..dims.[1] - 1 -> result.[0, i] |]|]
    {|Start = res.[0]; End = res.[1]|}

let qas = 
    formatforSquadQA (c, 
        "What will happen to Stack Overflow ?")
    |> Array.map (robertaEncoder robertaBin >> Array.head)
let answers = Array.map runrobertaQA qas

[|for n in 0..2 do 
    [|for (i,a) in Array.zip [|0..answers.Length - 1|] answers do
        let s, sp = nthMax n a.Start
        let e, ep = nthMax n a.End
        if exp sp > 1.f then
            let str = qas.[i].[s..e] |> robertaDecoder vocab
            if str <> "" && not(str.Contains "<s>")then 
                yield str, exp sp , exp ep|] 
   |]

BlingFireUtils.GetWords c |> Seq.toArray 
|> Array.countBy id 
|> Array.sortByDescending snd 

starts 
|> Array.map exp 
|> Array.normalize
|> Array.indexed
|> Prelude.TextHistogram.genericHistogram float 20 

