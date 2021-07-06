//#r "System.Runtime.InteropServices"  
#load "tokenizers.fsx"
#load "sampler.fsx"
#load "TestDocSnippets.fsx"
#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
#time "on"

open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 
open Prelude 
open BlingFire
open Newtonsoft.Json 
open ``Tensor-extensions``  
open Tokenizers 
open Sampler

let vocab =
    let vocabDict =
        JsonConvert.DeserializeObject<Collections.Generic.Dictionary<string, int>>
            (IO.File.ReadAllText
                 @"D:\Downloads\NeuralNets\roberta-base-squad2\vocab.json")
    [| for KeyValue(w, _) in vocabDict -> w |]

initBlingFire()

let robertaTokHandle = BlingFireUtils.LoadModel  @"C:/Users/cybernetic/source/repos/BlingFire/bin/Release/net5.0/Files/roberta.bin"
let bertTokHandle = BlingFireUtils.LoadModel  @"C:/Users/cybernetic/source/repos/BlingFire/bin/Release/net5.0/Files/bert_base_tok.bin"
let bertCasedTokHandle = BlingFireUtils.LoadModel  @"C:/Users/cybernetic/source/repos/BlingFire/bin/Release/net5.0/Files/bert_base_cased_tok.bin"

///////////
// allocate space for ids 
let ids = Array.create 1024 0

let inBytes = Text.Encoding.UTF8.GetBytes("hello this is a test</s>");

let _ = BlingFireUtils.TextToIds (robertaTokHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(),  ids.Length, 0)
let _ = BlingFireUtils.TextToIds (bertTokHandle, inBytes.AsSpan(), inBytes.Length, ids.AsSpan(),  ids.Length, 0)

ids

//Test sentence splitting
BlingFireUtils.GetSentences "This is a test of 3.45 people with Ph.D's said\n Mr. Robert. Who will take this."
|> Seq.toArray
|> printfn "%A"

/////////////////////////////

let startToken, stopToken = 0, 2

let appendStartEndTokens s = 
    [|startToken; yield! s; stopToken|]
    
//Renaming for semantic purposes
let robertaTokenizer (s: string) =
    generalTokenizer robertaTokHandle s 
         
let bertTokenizer (s: string) =
    generalTokenizer bertTokHandle s  

let bertCasedTokenizer (s: string) =
    generalTokenizer bertCasedTokHandle s  

let robertaDetokenizer (vocab:string []) tokens = 
    let inline lookUpFixHack (s:string) = s.Replace("âĢĻ", "’").Trim(' ')
    tokens 
    |> Seq.fold (fun str t -> 
        let w = vocab.[t] 
        if w.[0] = 'Ġ' then str + " " + w.[1..] 
        else str + w) ""
    |> lookUpFixHack 
     
//////////////////
bertTokenizer "A B"
//The blingfire encoder removes newlines. Transformer LMs are quite sensitive to whitespace
robertaTokenizer ("A\nB")

let toks = robertaTokenizer "I don't think that's ribonucleic"
        
robertaDetokenizer vocab toks

///Appending tokens test
let w1 = robertaTokenizer "Hello there"

appendStartEndTokens w1 |> robertaDetokenizer vocab

////////////////////

//The problem of decoding unicode characters. Using vocab.json seems to result in mangling and using a lookup table will not scale.
//How to fix? State: Don't know.

[|for t in robertaTokenizer "漢" do 
    yield! vocab.[t].Replace("Ġ", "").ToCharArray() |]
|> Array.map byte 
|> Strings.DecodeFromUtf8Bytes

Strings.toUTF8Bytes "漢"

[|for t in robertaTokenizer "’" do 
    yield! vocab.[t].Replace("Ġ", "").ToCharArray() |]
|> Array.map byte 
|> Strings.DecodeFromUtf8Bytes

[|for t in robertaTokenizer "Inside the apple’s" do 
    yield vocab.[t].Replace("Ġ", "")|]

Strings.toUTF8Bytes "’"

Strings.toUTF8Bytes "ë"

Array.map2 (-) [|229uy; 173uy; 151uy|] [|229uy; 67uy; 57uy|]

Array.map2 (-) [|226uy; 128uy; 153uy|] [|226uy; 34uy; 59uy|]

Array.map2 (-) [|226uy; 134uy; 144uy|] [|226uy; 40uy; 50uy|]

[|for t in robertaTokenizer "字" -> vocab.[t]|]

[|for t in robertaTokenizer "漢" -> vocab.[t]|]

"æ¼¢åŃĹ".ToCharArray() |> Array.map byte |> Strings.DecodeFromUtf8Bytes

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



//////////////////////////////////////
//Stats on token size vs words len 
let testWordsSplit (s:string) =
    {|Token = robertaTokenizer s |> Array.length; 
      Words = Strings.splitToWordsRegEx s |> Array.length |}


//////////////////
"I am going to the store. I bought a thing." |> robertaTokenizer 
    = ([|"I am going to the store."; "I bought a thing."|] 
         |> Array.collect robertaTokenizer)


let robertaTokenizerInfo = {StartToken = Some 0; StopToken = 2}

let robertaTokenizeSplit maxlen (sents : _ []) = 
    tokenizeSplit (tokenize robertaTokenizerInfo robertaTokenizer) maxlen sents
      
let testsents = BlingFireUtils.splitSentence TestDocSnippets.reals  

let maxlen = 512// = 0
 
robertaTokenizeSplit 25 [|""|]
robertaTokenizeSplit 25 [|"A"|]
robertaTokenizeSplit 12 testsents
robertaTokenizeSplit 15 testsents 
robertaTokenizeSplit 20 testsents 
robertaTokenizeSplit 32 testsents 
robertaTokenizeSplit 50 testsents 
robertaTokenizeSplit 512 testsents 

let stripTags (a:_[]) = a.[1..^1]

robertaTokenizeSplit 50 testsents
|> Array.collect stripTags = robertaTokenizer TestDocSnippets.reals

robertaTokenizeSplit 512 testsents 
|> Array.concat
|> stripTags = robertaTokenizer TestDocSnippets.reals 

Array.map robertaTokenizer testsents

Array.map (robertaTokenizer >> Array.length) testsents //'|> Array.sum

//////////////////////////

let tokenizeForSquadQA (context : string) (question : string) =
    let q =
        robertaTokenizeSplit 512 (BlingFireUtils.splitSentence question) 
        |> Array.head

    context
    |> BlingFireUtils.splitSentence
    |> robertaTokenizeSplit (512 - q.Length)
    |> Array.map (fun c -> Array.append q c)
     
let runrobertaQA (robertaQA:InferenceSession) (tokens:int []) =
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|tokens|])
    
    let outputs = robertaQA.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    let res =
        [| for output in outputs -> 
            let result = output.AsTensor<float32>()
            let vec = result.[0, *]      
            output.Dispose()
            vec |]
        
    outputs.Dispose()
    
    {|Start = res.[0]; End = res.[1]|}


let questionAsk robertaQA context question = 
    let qas = tokenizeForSquadQA context question

    printfn "%A" (Array.map Array.length qas)

    let answers = Array.map (runrobertaQA robertaQA) qas

    [| for n in 0 .. 2 do
        for (i, ans) in Array.zip [| 0 .. answers.Length - 1 |] answers do
            let start, sprob = Array.nthTop n ans.Start
            let endIndex, eprob = Array.nthTop n ans.End

            if exp sprob > 1.f then
                let str =
                    qas.[i].[start..endIndex]
                    |> robertaDetokenizer vocab

                if str <> "" && not (str.Contains "<s>") then
                    yield
                        {|  Snippet = str
                            StartProb = exp sprob
                            EndProb = exp eprob
                            nth = n
                            Para = i |}  |]

////////////////////
///////////////////////// 

let robertaQA = new InferenceSession(@"D:\Downloads\NeuralNets\roberta-base-squad2\roberta-base-squad2.onnx")

robertaQA.OutputMetadata

let context, question = ("I picked an apple, ate it then went next door to pick a banana.", "What door did I go through ?")
let qas = tokenizeForSquadQA context question
let answers = Array.map (runrobertaQA robertaQA) qas

answers.[0].Start 
|> Array.map exp 
|> Array.normalize
|> Array.mapi (fun i p -> robertaDetokenizer vocab [qas.[0].[i]], p)
|> TextHistogram.genericHistogram float 20 

//////////////////////////////

 
let str = IO.File.ReadAllText (pathCombine DocumentsFolder "doc.txt")

let sents = BlingFireUtils.splitSentence str 
 
questionAsk robertaQA str "What does RoseTTAFold do ?"
|> Array.sortBy (fun a -> a.Para)


BlingFireUtils.GetWords str
|> Seq.toArray
|> Array.countBy id
|> Array.sortByDescending snd

//////////////////
let str  = "The number r, which is the number of one-forms that the tensor T eats, is called the contravariant degree and the number s, which is the number of vectors the tensor T eats, is called the covariant degree."
 
let qs = ["How can the problem with the reals be fixed?"; "what is the problem?"; "what is the goal"; "How can the problem be fixed?"]

/////////////////////////
testWordsSplit str

[3285./3879.; 637. / 792.; 664. / 832.; 7345./9748.]

//Using just space
[581./727.; 788./1032.;244./322.; 530./741.; 565./771.; 2073./ 3270. ]
/////////////////////// 

let bartModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-encoder.onnx")

let bartModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\bart-large-cnn\bart-large-cnn-decoder.onnx")

let tok0 = robertaTokenizer "This is a test." 

let tok0 = [|[|152; 16; 10; 1296; 1437|]|]
 
let toks = robertaTokenizeSplit 512 sents  

let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|toks.[0]|])

t

robertaDetokenizer vocab tok0.[0]

let outputs = bartModelEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]

let encoderStates = Seq.head outputs

encoderStates.Name <- "encoder_hidden_states"
   
outputs.Dispose()

encoderStates.Dispose()  


robertaDetokenizer vocab []

decoderSampler bartModelDecoder Array.argmaxf32 robertaTokenizerInfo.StopToken 50 encoderStates


////////////////////////////
//The bart models are resource hogs, slow, sometimes generate summaries that are highly misleading to outright wrong and I cannot seem to get model export
//to not give inferences with extremely high entropy. If the probability mass is not so concentrated that greedy sampling works 
//then the model is too impractical for interactive use. I'll not be using them.

(**UPDATE: It looks like I actually didn't initialize generation with a start token and so some of the problems I was seeing was due to that. 
Having put the start token back, the results were much better but it still takes a lot of time to compute and the results aren't that great; 
sometimes it still makes things up and there still seems to be...
I probably did not properly convert the BART weights to onnx because the output distribution still seems too high entropy *)

let distilbartModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\distilbart-cnn-6-6\distilbart-cnn-6-6-encoder.onnx")

let distilbartModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\distilbart-cnn-6-6\distilbart-cnn-6-6-decoder.onnx")

let tokens = robertaTokenizeSplit 1024 sents  

let ss1 =
    nnSampler bartModelEncoder bartModelDecoder (robertaDetokenizer vocab)
        robertaTokenizerInfo.StopToken Array.argmaxf32 200 tokens 

let ss2 =
    nnSampler distilbartModelEncoder distilbartModelDecoder
        (robertaDetokenizer vocab) robertaTokenizerInfo.StopToken
        Array.argmaxf32 200 tokens 

ss1

ss2

//////////////////////////

let enc = new InferenceSession(@"D:\Downloads\NeuralNets\ctrlsum-arxiv\ctrlsum-arxiv-encoder.onnx") 
let dec = new InferenceSession(@"D:\Downloads\NeuralNets\ctrlsum-cnndm\ctrlsum-cnndm-decoder.onnx") 


let tokenizeforCtrlSummary (context : string) =
    
    let contextCommand = robertaTokenizer "loopy => "
    
    context
    |> BlingFireUtils.splitSentence
    |> robertaTokenizeSplit (1024 - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [contextCommand; c])


let ctokens = tokenizeforCtrlSummary str  

nnSampler enc dec
    (robertaDetokenizer vocab) robertaTokenizerInfo.StopToken
    (Array.argmaxf32) 200 tokens 

