
#load "tokenizers.fsx"
#load "sampler.fsx"
#load "TestDocSnippets.fsx"
#time "on"
#r @"SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net472\SentencePieceDotNET.dll" 

open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 
open ``Tensor-extensions``  
open Tokenizers 
open Sampler

let spiece = new SentencePieceDotNET.SentencePieceDotNET()

spiece.Load(@"D:\Downloads\NeuralNets\t5-base\spiece.model")
 

let t5TokenizerInfo = {StartToken = None; StopToken = 1}
 
let t5Tokenizer (s: string) = spiece.Encode(s)

let t5detokenizer ids = 
    let ids' = Array.filter ((>) 32_000) ids //remove special tokens which crash our implementation
    if ids'.Length > 0 then spiece.Decode ids'
    else ""

let t5TokenizeSplit maxlen (sents : _ []) =   
    tokenizeSplit (tokenize t5TokenizerInfo t5Tokenizer) maxlen sents



let str = IO.File.ReadAllText (pathCombine DocumentsFolder "doc.txt")

let sents = BlingFireUtils.splitSentence str 

       
let t5tokens = t5TokenizeSplit 512 sents


let t5ModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-small\t5-small-encoder-quantized.onnx")

let t5ModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-small\t5-small-decoder-quantized.onnx")

let t5SummarizerEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-encoder.onnx")
let t5SummarizerDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-decoder.onnx")

let t5largeEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-large\t5-large-encoder.onnx")
let t5largeDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-large\t5-large-decoder.onnx")

let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|t5tokens.[0]|])
let t = Tensors.ArrayTensorExtensions.ToTensor(array2D (t5TokenizeSplit 512 [|"This is a test"|]))


t5largeEncoder.Dispose()

Tensor.toArray2D t

t5detokenizer t5tokens.[1]


let outputs = t5ModelEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]

let encoderStates = Seq.head outputs

encoderStates.Name <- "encoder_hidden_states"
   
outputs.Dispose()

encoderStates.Dispose()  
 
encoderStates.AsTensor<float32>() |> Tensor.toMatrix 
 

let tokenizeforT5Summary (context : string) =
    
    let contextCommand = t5Tokenizer "summarize:"
    
    context
    |> BlingFireUtils.splitSentence
    |> t5TokenizeSplit (512 - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [contextCommand; c])

let tokenizeforT5QA (context : string) (question : string) =
    let q =
        t5Tokenizer ("question: " + question)
        |> Array.takeOrMax 511
    
    let contextCommand = t5Tokenizer "context:"
    
    context
    |> BlingFireUtils.splitSentence
    |> t5TokenizeSplit (512 - q.Length - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [q; contextCommand; c])


let summ = nnSampler t5SummarizerEncoder t5SummarizerDecoder t5detokenizer t5TokenizerInfo.StopToken (Array.argmaxf32) 150 t5tokens

let summ = nnSampler t5ModelEncoder t5ModelDecoder t5detokenizer t5TokenizerInfo.StopToken (Array.argmaxf32 ) 200 t5tokens

let summ = nnSampler t5largeEncoder t5largeDecoder t5detokenizer t5TokenizerInfo.StopToken (Array.argmaxf32 ) 200 t5tokens

t5SummarizerDecoder.Dispose()

let str2 = 
    summ |> String.concat "\n"

let c = t5Tokenizer "This is a test"
let q = t5Tokenizer "summarize: "

t5Tokenizer str
t5Tokenizer "question"
t5Tokenizer "apple"

t5Tokenizer "question apple" |> t5detokenizer
 
Array.append q c = t5Tokenizer "summarize: This is a test"

t5Tokenizer "summarize: This is a test" = (Array.head (tokenizeforT5Summary "This is a test")).[..^1]
 
tokenizeforT5QA "This is a house" "Where are we ?"

let t5tokens = tokenizeforT5QA str "What does RoseTTAFold do?"
t5tokens.Length
let t5tokens = tokenizeforT5Summary str

let testWordsSplitT5 (s:string) =
    {|Token = t5Tokenizer s |> Array.length; 
      Words = Strings.splitToWordsRegEx s |> Array.length |}

testWordsSplitT5 str

[664./895.; 7345./10987.]
